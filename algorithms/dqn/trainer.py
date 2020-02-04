
import os
import math
import random
import time
from itertools import count
import signal

import numpy as np
import torch
import torch.nn.functional as F
from apex import amp

import utils.logx
from algorithms.dqn.utils import replay_mem
from utils import gym_utils
from utils import visualization


class DQNTrainer(object):
    def __init__(self, train_cfg, env, agent, target_net, policy_net, memory, optimizer,
                 num_episodes, device, env_random_state_list, env_initial_state_screen_list):

        self.cfg = train_cfg
        self.agent = agent
        self.env = env
        self.optimizer = optimizer
        self.target_net = target_net
        self.policy_net = policy_net
        self.memory = memory
        self.steps_done = 0
        self.init_episode = 0
        self.num_episodes = num_episodes
        self.device = device
        self.episode_durations = []
        self.estimated_validation_score_list = []
        self.empirical_validation_score_list = []
        self.env_random_state_list = env_random_state_list
        self.env_initial_state_screen_list = env_initial_state_screen_list

        self.logger = utils.logx.EpochLogger(
            self.cfg.LOG.OUTPUT_DIR,
            self.cfg.LOG.OUTPUT_FNAME,
            self.cfg.LOG.EXP_NAME
        )

        self._load_ckpt(self.cfg.CKPT_PATH)

    def train(self):
        start_time = time.time()

        estimated_validation_episodes_list = []
        empirical_validation_episodes_list = []
        for i_episode in range(self.init_episode, self.num_episodes):
            self.curr_episode = i_episode
            self._graceful_exit()

            # Initialize the environment and state.
            self.env.reset()
            last_screen = gym_utils.get_screen(self.env).to(self.device)
            current_screen = gym_utils.get_screen(self.env).to(self.device)
            self.agent.state = current_screen - last_screen

            self._train_episode(current_screen)
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.cfg.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.agent.policy_net.state_dict())

            if i_episode % self.cfg.ESTIMATED_VALIDATION_FREQUENCY == 0:
                validation_score = self.validate(val_type='q_value')
                self.estimated_validation_score_list.append(validation_score)
                estimated_validation_episodes_list.append(i_episode)
                if self.cfg.VISUALIZE:
                    visualization.plot_validation_score(self.estimated_validation_score_list,
                                                        estimated_validation_episodes_list,
                                                        fig_num=3, y_label='Q value')

            if (i_episode % self.cfg.EMPIRICAL_VALIDATION_FREQUENCY == 0) and (i_episode > 0):
                validation_score = self.validate(val_type='score')
                self.empirical_validation_score_list.append(validation_score)
                empirical_validation_episodes_list.append(i_episode)
                if self.cfg.VISUALIZE:

                    visualization.plot_validation_score(self.empirical_validation_score_list,
                                                        empirical_validation_episodes_list,
                                                        fig_num=4, y_label='Duration')


            if i_episode % self.cfg.CKPT_SAVE_FREQ == 0:
                self._save_ckpt(i_episode)

    def _train_episode(self, current_screen):
        """

        :param current_screen:
        :return:
        """
        state_history = []
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
            current_screen = gym_utils.get_screen(self.env).to(self.device)
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
                episode_duration = t + 1
                self.episode_durations.append(episode_duration)
                if self.cfg.VISUALIZE:
                    visualization.plot_durations(self.episode_durations)
                return

    def step(self):

        if len(self.memory) < self.cfg.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.cfg.BATCH_SIZE)
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

        if self.cfg.OPT_LEVEL == "O0":
            data_type = torch.float
        else:
            data_type = torch.half

        next_state_values = torch.zeros(self.cfg.BATCH_SIZE, device=self.device,
                                        dtype=data_type)

        next_state_values[non_final_mask] = \
            self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.cfg.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()

        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()

        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def validate(self, val_type='q_value'):
        validation_values = []
        if val_type == 'q_value':
            for state in self.env_random_state_list:
                current_state_q = max(
                    self.policy_net(state.to(self.device)).data.cpu().numpy()[0])
                validation_values.append(current_state_q)
            validation_value = np.mean(validation_values)
        elif val_type == 'score':
            for state, screen in self.env_initial_state_screen_list:
                # Initialize the state.
                self.env.reset()
                self.agent.state = state
                _, episode_duration = self.agent._play_episode(current_screen=screen)
                validation_values.append(episode_duration)
            validation_value = np.mean(validation_values)
        return validation_value

    def _save_ckpt(self, episode=None):
        checkpoint = {
            'model': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': amp.state_dict(),
            'episode': episode,
            'steps_done': self.steps_done,
            'init_episode': self.init_episode
        }
        fpath = 'checkpoint_' + \
                ('%d' % episode if episode is not None else '0') + '.pt'
        checkpoint_path = os.path.join(self.cfg.CKPT_SAVE_DIR, fpath)
        torch.save(checkpoint, checkpoint_path)

    def _load_ckpt(self, ckpt_path):
        newest_ckpt_name = self._get_newest_ckpt()

        if newest_ckpt_name is not None:
            ckpt_path = os.path.join(
                self.cfg.CKPT_SAVE_DIR, newest_ckpt_name)

        if ckpt_path != '':
            print('Loading {}'.format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            self.policy_net.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            self.steps_done = checkpoint['steps_done']
            self.init_episode = checkpoint['init_episode']

    def _graceful_exit(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print('Recieved signal {}.'.format(signum))
        print('Saving checkpoint for episode {}'.format(self.curr_episode))
        self._save_ckpt(self.curr_episode)
        exit(0)

    def _get_newest_ckpt(self):
        files = os.listdir(self.cfg.CKPT_SAVE_DIR)
        newest_ckpt_name = None
        if files != []:
            newest_ckpt_name = max(files, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        return newest_ckpt_name


class DQNAgent(object):
    def __init__(self, policy_net, n_actions, device, env, epsilon=0.05):
        self.policy_net = policy_net
        self.state = None
        self.n_actions = n_actions
        self.device = device
        self.env = env
        self.epsilon = epsilon

    def _play_episode(self, current_screen):
        """

        :param current_screen:
        :param return_state_history:
        :param validation:
        :return:
        """
        states = []
        for t in count():

            # Select and perform an action
            eps_threshold = self.epsilon
            action = self.select_action(eps_threshold)
            # self.steps_done += 1
            _, _, done, _ = self.env.step(action.item())

            states.append(current_screen)

            last_screen = current_screen
            current_screen = gym_utils.get_screen(self.env).to(self.device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            # self.memory.push(self.agent.state, action, next_state, reward)

            # Move to the next state
            self.state = next_state

            # Perform one step of the optimization (on the target network)
            # self.step()

            if done:
                episode_duration = t + 1
                return states, episode_duration


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
