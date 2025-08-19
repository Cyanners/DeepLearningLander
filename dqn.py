import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        neuronz = 64
        # self.fc1 = nn.Linear(state_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, action_dim)
        # self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(state_dim, neuronz)
        self.fc2 = nn.Linear(neuronz, neuronz)
        self.fc3 = nn.Linear(neuronz, neuronz // 2)
        self.fc4 = nn.Linear(neuronz // 2, action_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent:
    def __init__(self, device, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.998, target_update_freq=100):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.memory = deque(maxlen=50000)  # Increased memory
        self.batch_size = 64

        self.device = device

        # Main network and target network
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.model(states).gather(1, actions)

        # Standard DQN
        # with torch.no_grad():
        #     q_next = self.target_model(next_states).max(1)[0].unsqueeze(1)
        #     q_target = rewards + self.gamma * q_next * (1 - dones)

        # Double DQN target Q values
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1).unsqueeze(1)  # action selection from online net
            q_next = self.target_model(next_states).gather(1, next_actions)  # value from target net
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()