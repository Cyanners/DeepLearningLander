import pygame
import torch
import random
from env import LunarLanderEnv
from dqn import DQNAgent

landed_bool = False

# Toggle between training and replay
replay = False

env = LunarLanderEnv(800, 600, 5)

# state_dim = 6 # x, y, vx, vy, terrain_range, above_pad
state_dim = 5 # vx, vy, terrain_range, above_pad, pad_dist_x
action_dim = 4 # none, up, left, right

agent = DQNAgent(state_dim, action_dim)

if replay:
    # Load pre-trained weights
    agent.model.load_state_dict(torch.load("dqn_lander.pth"))
    agent.model.eval()  # put network in evaluation mode



while not landed_bool:

    # -----------------------------------------
    # Training Mode
    # -----------------------------------------

    if not replay:

        num_episodes = 2000
        max_steps = 3000

        agent.epsilon = 1.0

        for episode in range(num_episodes):

            # Show rendering only every 100 episodes
            if episode % 100 == 12345:
                env.set_rendering(True)
            else:
                env.set_rendering(False)

            state = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:

                # Events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # # Determine action from keys
                # keys = pygame.key.get_pressed()
                # action = 0  # 0 = no thrust, 1 = up, 2 = left, 3 = right
                # if keys[pygame.K_UP]:
                #     action = 1
                # elif keys[pygame.K_LEFT]:
                #     action = 2
                # elif keys[pygame.K_RIGHT]:
                #     action = 3



                # Select action with DQNAgent
                action = agent.select_action(state)

                # Step in environment
                next_state, reward, done, info = env.step(action)
                total_reward += reward

                # Store transition
                agent.memory.append((state, action, reward, next_state, float(done)))

                # Train the agent
                agent.train_step()

                state = next_state



                if episode % 100 == 12345:
                    env.render(text_wait=True)
                    env.clock.tick(60)

                steps += 1
                if steps == max_steps:
                    done = True

                if done:
                    # State: self.x, self.y, self.vx, self.vy, self.terrain_range, self.above_pad
                    # State: self.vx, self.vy, self.terrain_range, self.above_pad, self.pad_dist_x
                    print(f"Episode {episode+1}:\t", end="")
                    if not info['landed'] and not info['crashed']:
                        print(f"FLYING, Fuel: {env.lander.fuel:.2f}%, {info['elapsed_ms']}ms, ", end="")
                    elif info['landed']:
                        print(f"LANDED ({info['landing_accuracy']:.2f}%), Fuel: {env.lander.fuel:.2f}%, {info['elapsed_ms']}ms, ", end="")
                    elif info['crashed']:
                        print(f"CRASHED, Fuel: {env.lander.fuel:.2f}%, {info['elapsed_ms']}ms, ", end="")

                    print(f"Rwrd: {total_reward:.2f}, e: {agent.epsilon:.3f}")
                    # print(f"\tx:{state[0]:.2f}, y:{state[1]:.2f}, vx:{state[2]:.2f}, vy:{state[3]:.2f}")

                    if info['landed']:
                        landed_bool = True
                        break

            if landed_bool:
                break

            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay


        # All episodes done -------------------------------------------------------------------

        # Save the trained model
        torch.save(agent.model.state_dict(), "dqn_lander.pth")


    # -----------------------------------------
    # Replay Mode
    # -----------------------------------------

    else:
        print("Replaying with trained agent...")
        while True:
            env.set_rendering(True)  # always render in replay
            state = env.reset()
            done = False

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                epsilon = 0.0  # exploration rate (10%)

                if random.random() < epsilon:
                    # Explore: random action
                    action = random.randint(0, agent.action_dim - 1)
                else:
                    # Exploit: greedy action
                    with torch.no_grad():
                        q_values = agent.model(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                        action = torch.argmax(q_values).item()

                state, reward, done, info = env.step(action)
                env.render(text_wait=True)
                env.clock.tick(60)

            # Print outcome each attempt
            if info['landed']:
                print(f"LANDED ({info['landing_accuracy']:.2f}%), Fuel left: {state[4]:.2f}%")
            elif info['crashed']:
                print("CRASHED!")
            else:
                print("TIMEOUT!")

pygame.quit()

# 3 Metrics of success:
#   Time elapsed - the faster the lander lands, the better
#   Fuel used - the less fuel the lander uses, the better
#   Accuracy - the more accurate the landing position, the better
# These should be somehow combined into a weighted score...maybe?