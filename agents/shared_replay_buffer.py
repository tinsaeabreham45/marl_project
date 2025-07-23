import random
from collections import deque
import numpy as np

class SharedReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, message, fingerprint):
        self.buffer.append((state, action, reward, next_state, done, message, fingerprint))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, messages, fingerprints = zip(*batch)
        return (
            np.array(states),
            list(actions),
            list(rewards),
            np.array(next_states),
            list(dones),
            list(messages),
            np.array(fingerprints)
        )

    def __len__(self):
        return len(self.buffer)
