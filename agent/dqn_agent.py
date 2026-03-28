import torch
import random
import torch.nn as nn
import torch.optim as optim
from model.dqn_model import DQN

class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)

        self.gamma = config.GAMMA

        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY

    def act(self, state, env):
        if random.random() < self.epsilon:
            return env.action_space.sample()

        with torch.no_grad():
            state = torch.FloatTensor(state) / 10   # 🔥 normalization
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def train(self, memory, batch_size):
        # 🔥 warmup (very important)
        if memory.size() < 1000:
            return

        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        states = states / 10
        next_states = next_states / 10

        # Current Q
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q (using target network)
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)