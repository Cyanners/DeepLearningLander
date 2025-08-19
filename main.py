import pygame
import torch
import random
import numpy as np
from torch import optim
from env import LunarLanderEnv
from dqn import DQNAgent
from tqdm import tqdm, trange

# Toggle between training and replay
replay = True

if replay:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

state_dim = 7
action_dim = 4 # none, up, left, right

# env = LunarLanderEnv(800, 600, 5)
env = LunarLanderEnv(800, 600, random.randint(1, 1000))

agent = DQNAgent(device, state_dim, action_dim)

if replay:
    # Load pre-trained weights
    agent.model.load_state_dict(torch.load("weights/dqn_lander_ep2808.pth"))
    agent.model.eval()  # put network in evaluation mode


training_done = False
while not training_done:

    # -----------------------------------------
    # Training Mode
    # -----------------------------------------

    if not replay:

        # Track performance
        episode_rewards = []
        success_rate = []
        losses = []

        num_episodes = 3000
        max_steps = 2000

        start_pos = (0, 0)
        agent.epsilon = 1.0

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(agent.optimizer, step_size=500, gamma=0.8)

        progress_bar = tqdm(range(num_episodes), desc="Episodes")

        for episode in progress_bar:

            # Vary terrain seed for better generalization
            if episode % 100 == 0:
                env = LunarLanderEnv(800, 600, random.randint(1, 1000))
                start_pos = (random.randint(50, 750), random.randint(50, 150))

            # Show rendering only every 100 episodes
            if episode % 100 == 12345:
                env.set_rendering(True)
            else:
                env.set_rendering(False)

            state = env.reset(start_pos[0], start_pos[1])
            done = False
            total_reward = 0
            steps = 0
            episode_losses = []

            while not done and steps < max_steps:

                # Events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # Select action with DQNAgent
                action = agent.select_action(state)

                # Step in environment
                next_state, reward, done, info = env.step(action)
                total_reward += reward

                # Store transition
                agent.store_transition(state, action, reward, next_state, done)

                # Train the agent (multiple times per step for better sample efficiency)
                if len(agent.memory) > agent.batch_size:
                    for _ in range(2):  # Train multiple times per step
                        loss = agent.train_step()
                        if loss is not None:
                            episode_losses.append(loss)

                state = next_state
                steps += 1

                if episode % 100 == 12345:
                    env.render(text_wait=True)
                    env.clock.tick(60)

            # Single episode done ---------------------------------------------------------------

            episode_rewards.append(total_reward)
            if episode_losses:
                losses.append(np.mean(episode_losses))

            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            scheduler.step()

            # Track success rate over last 100 episodes
            if episode >= 100:
                recent_episodes = episode_rewards[-100:]
                success_count = sum(1 for r in recent_episodes if r > 50)  # Adjust threshold
                success_rate.append(success_count / 100)

            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else total_reward
                avg_loss = np.mean(losses[-50:]) if len(losses) >= 50 else 0
                current_success_rate = success_rate[-1] if success_rate else 0
                # Update tqdm status line
                progress_bar.set_description_str(f"Ep {episode} | R: {avg_reward:.1f} | SR: {current_success_rate:.2%} | ε: {agent.epsilon:.3f} | L: {avg_loss:.1f}")
                print()

            # Save model periodically and when performance improves
            if success_rate and success_rate[-1] > 0.9:
                torch.save(agent.model.state_dict(), f"weights/dqn_lander_ep{episode}.pth")


    # -----------------------------------------
    # Replay Mode
    # -----------------------------------------

    else:
        print("Replaying with trained agent...")
        while True:
            env = LunarLanderEnv(800, 600, random.randint(1, 1000))
            env.set_rendering(True)  # always render in replay
            start_pos = (random.randint(50, 750), random.randint(50, 150))
            state = env.reset(start_pos[0], start_pos[1])
            done = False
            total_reward = 0

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                epsilon = 0.02  # exploration rate

                if random.random() < epsilon:
                    # Explore: random action
                    action = random.randint(0, agent.action_dim - 1)
                else:
                    # Exploit: greedy action
                    with torch.no_grad():
                        q_values = agent.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                        action = torch.argmax(q_values).item()

                state, reward, done, info = env.step(action)
                total_reward += reward

                env.render(text_wait=True)
                env.clock.tick(60)

            # Print outcome each attempt
            if not info['landed'] and not info['crashed']:
                print(f"FLYING, Fuel: {env.lander.fuel:.2f}%, {info['elapsed_ms']}ms, ", end="")
            elif info['landed']:
                print(f"LANDED ({info['landing_accuracy']:.2f}%), Fuel: {env.lander.fuel:.2f}%, {info['elapsed_ms']}ms, ", end="")
            elif info['crashed']:
                print(f"CRASHED, Fuel: {env.lander.fuel:.2f}%, {info['elapsed_ms']}ms, ", end="")

            print(f"Rwrd: {total_reward:.2f}, e: {agent.epsilon:.3f}")

pygame.quit()

# 3 Metrics of success:
#   Time elapsed - the faster the lander lands, the better
#   Fuel used - the less fuel the lander uses, the better
#   Accuracy - the more accurate the landing position, the better
# These should be somehow combined into a weighted score...maybe?