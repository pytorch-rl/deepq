import random
from itertools import count

import torch


class DQNAgent():
    def __init__(self, policy_net, n_actions, device, env, epsilon=0.0):
        self.policy_net = policy_net
        self.state = None
        self.n_actions = n_actions
        self.device = device
        self.env = env
        self.epsilon = epsilon

    def _play_episode(self):
        """A single episode (until agent fails the game), without optimizing
        policy net.

        Returns:
            states: list of game states from the episode played (required for
                    generating validation set)
            episode_duration: number of steps in the episode played
        """
        states = []
        for t in count():
            states.append(self.state)

            # Select and perform an action
            action = self.select_action(eps_threshold=0)
            _, _, done, _ = self.env.step(action.item())

            if not done:
                self.state = self.env.get_state().to(self.device)

            if done:
                episode_duration = t + 1
                return states, episode_duration

    def select_action(self, eps_threshold):
        """Selection of action to play

        Args:
            eps_threshold: exploration (random action) probability

        Returns:
            action: chosen action to play

        """
        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(self.state).max(1)[1].view(1, 1)

        else:
            action = torch.tensor([[random.randrange(self.n_actions)]],
                                  device=self.device,
                                  dtype=torch.long)

        return action
