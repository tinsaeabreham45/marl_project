import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size + 2, 128),  # obs_size is 18, +1 for message, +1 for fingerprint
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, obs_size, action_size, shared_buffer, lr=1e-4, gamma=0.99):
        self.q_net = QNetwork(obs_size, action_size)
        self.target_net = QNetwork(obs_size, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.shared_buffer = shared_buffer
        self.gamma = gamma
        self.batch_size = 64
        self.action_size = action_size

    def act(self, obs, epsilon=0.1, message=0.0, fingerprint=0.0):
        obs_with_msg_fp = np.append(obs, [message, fingerprint])
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        obs_tensor = torch.FloatTensor(obs_with_msg_fp).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done, message, fingerprint):
        self.shared_buffer.add(state, action, reward, next_state, done, message, fingerprint)

    def learn(self):
        if len(self.shared_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, messages, fingerprints = self.shared_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        messages = torch.FloatTensor(messages).unsqueeze(1)
        fingerprints = torch.FloatTensor(fingerprints).unsqueeze(1)
        states_with_msg_fp = torch.cat([states, messages, fingerprints], dim=1)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        next_messages = torch.zeros_like(messages)
        next_fingerprints = torch.zeros_like(fingerprints)
        next_states_with_msg_fp = torch.cat([next_states, next_messages, next_fingerprints], dim=1)

        # DDQN: Use primary network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_net(next_states_with_msg_fp).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states_with_msg_fp).gather(1, next_actions)
        target = rewards + self.gamma * next_q_values * (1 - dones)
        q_values = self.q_net(states_with_msg_fp).gather(1, actions)
        loss = self.criterion(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())