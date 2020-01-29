import time
import random
import math
from utils import gym_utils
from utils import visualization
from itertools import count
import torch
import torch.nn.functional as F
from config.default_config import cfg
from algorithms.dqn.utils import replay_mem

import utils.logx


class DQNTrainer(object):
    def __init__(self, train_cfg, env, agent, target_net, memory, optimizer, num_episodes, device):

        self.cfg = train_cfg
        self.agent = agent
        self.env = env
        self.optimizer = optimizer
        self.target_net = target_net
        self.memory = memory
        self.steps_done = 0
        self.num_episodes = num_episodes
        self.device = device
        self.episode_durations = []

        self.logger = utils.logx.EpochLogger(train_cfg.LOG.OUTPUT_DIR,
                                             train_cfg.LOG.OUTPUT_FNAME,
                                             train_cfg.LOG.EXP_NAME)

    def train(self):
        self.logger.setup_pt_saver()

        start_time = time.time()
        for i_episode in range(self.num_episodes):
            # Initialize the environment and state
            self.env.reset()
            last_screen = gym_utils.get_screen(self.env, self.device)
            current_screen = gym_utils.get_screen(self.env, self.device)
            self.agent.state = current_screen - last_screen

            for t in count():
                # Select and perform an action
                eps_threshold = self.cfg.EPS_END + (self.cfg.EPS_START - self.cfg.EPS_END) \
                                * math.exp(-1. * self.steps_done / self.cfg.EPS_DECAY)

                action = self.agent.select_action(eps_threshold)
                self.steps_done += 1
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
                self.memory.push(self.agent.state, action, next_state, reward)

                # Move to the next state
                self.agent.state = next_state

                # Perform one step of the optimization (on the target network)
                self.step()

                if done:
                    self.episode_durations.append(t + 1)
                    visualization.plot_durations(self.episode_durations)
                    break

                # TODO(maors): Log example.
                self.logger.store(X=t)
                self.logger.store(Y=i_episode)
                self.logger.store(Q=0.0)

            # Update the target network, copying all weights and biases in DQN.
            if i_episode % cfg.TRAIN.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.agent.policy_net.state_dict())

            # Save model.
            if (i_episode % self.cfg.LOG.SAVE_FREQ == 0) or (i_episode == i_episode - 1):
                self.logger.save_state()

            # Log info about epoch.
            self.logger.log_tabular('X')
            self.logger.log_tabular('Y')
            self.logger.log_tabular('Q')
            self.logger.log_tabular('Time', time.time() - start_time)
            self.logger.dump_tabular()

    def step(self):

        if len(self.memory) < cfg.TRAIN.BATCH_SIZE:
            return

        transitions = self.memory.sample(cfg.TRAIN.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = replay_mem.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(cfg.TRAIN.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = \
            self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * cfg.TRAIN.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def validate(self):
        pass


class DQNAgent(object):
    def __init__(self, policy_net, n_actions, device):
        self.policy_net = policy_net
        self.state = None
        self.n_actions = n_actions
        self.device = device

    def select_action(self, eps_threshold):
        sample = random.random()

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(self.state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]],
                                device=self.device,
                                dtype=torch.long)
