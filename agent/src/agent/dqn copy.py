import math
import random
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(name)s] (%(levelname)s) %(message)s')

junho_logger = logging.getLogger('Junho')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

GAMMA = 0.99


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 32
RMEM_CAPACITY = 10000


class DQN:
    def __init__(self, num_states, num_actions):
        self.step = 0
        self.episode = 0
        self.num_actions = num_actions

        self.memory = ReplayMemory(RMEM_CAPACITY)
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + GAMMA * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)

        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])

        return action

    def update_q_function(self):
        self.replay()

    def get_action(self, state):
        action = self.decide_action(state, self.episode)
        self.step += 1
        return action

    def memorize(self, state, action, state_next, reward):
        self.memory.push(state, action, state_next, reward)

    def reset(self):
        self.step = 0
        self.episode += 1
