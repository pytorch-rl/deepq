from collections import namedtuple
import random

from sortedcontainers import SortedList


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

SortedTransition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mean_dur'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SortedReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = SortedList(key=lambda x: x.mean_dur)
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.add(SortedTransition(*args))
        else:
            del self.memory[self.position]
            self.memory.add(SortedTransition(*args))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        lo = int(0.3 * len(self.memory))
        return random.sample(self.memory[-lo:], batch_size)

    def __len__(self):
        return len(self.memory)
