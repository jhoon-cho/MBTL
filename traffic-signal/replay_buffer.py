from collections import deque
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, memory_size_max, memory_size_min):
        self.buffer = deque(maxlen=memory_size_max)
        self.memory_size_max = memory_size_max
        self.memory_size_min = memory_size_min

    def push(self, state, action, reward, next_state):
        # push a record to the buffer
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        # get a batch of samples from the buffer
        if batch_size > self.size_now():
            state, action, reward, next_state = zip(*random.sample(self.buffer, self.size_now()))
            return np.concatenate(state), action, reward, np.concatenate(next_state)
        else:
            state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
            return np.concatenate(state), action, reward, np.concatenate(next_state)

    def size_now(self):
        return len(self.buffer)