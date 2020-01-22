import random
import math

import torch


class DQNTrainer(object):
    def __init__(self, train_cfg, env, agent, target_net, memory, optimizer):
        self.cfg = train_cfg
        self.agent = agent
        self.env = env
        self.optimizer = optimizer
        self.target_net = target_net
        self.memory = memory

    def train(self):
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            self.env.reset()
            last_screen = gym_utils.get_screen(self.env, self.device)
            current_screen = gym_utils.get_screen(self.env, self.device)
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                action, steps_done = self.agent.select_action()
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                last_screen = current_screen
                current_screen = gym_utils.get_screen(self.env, self.device)
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.step()

                # optimize_model(memory, policy_net, target_net, optimizer,
                #                device)
                #

                if done:
                    episode_durations.append(t + 1)
                    visualization.plot_durations(episode_durations)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % cfg.TRAIN.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

    def step(self):
        pass

    def validate(self):
        pass


EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


class DQNAgent(object):
    def __init__(self, state, policy_net, n_actions, device):
        self.policy_net = policy_net
        self.state = None
        self.steps_done = 0
        self.n_actions = n_actions
        self.device = device

    def select_action(self):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)

        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(self.state).max(1)[1].view(1, 1), self.steps_done
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],
                                device=self.device,
                                dtype=torch.long), self.steps_done
