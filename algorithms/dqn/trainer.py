import math
import time
from itertools import count

import numpy as np
import os
import random
import signal
import torch
import torch.nn.functional as F
import utils.logx
from algorithms.dqn.utils import replay_mem

from utils import visualization


# from apex import amp


class DQNTrainer(object):
    def __init__(self, train_cfg, env, agent, target_net, policy_net, memory, optimizer,
                 num_episodes, device, scheduler, env_random_states, env_initial_states):
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
        self.episode_mean_losses = []
        self.q_validation_scores = []
        self.score_validation_scores = []
        self.env_random_states = env_random_states
        self.env_initial_states_screens = env_initial_states
        self.scheduler = scheduler
        self.scheduler_steps = 0
        self.metric = -1
        self.episodes_from_scheduler_step = 0

        self._load_ckpt(self.cfg.CKPT_PATH)

        self.logger = utils.logx.EpochLogger(
            self.cfg.LOG.OUTPUT_DIR,
            self.cfg.LOG.OUTPUT_FNAME,
            self.cfg.LOG.EXP_NAME,
            self.cfg.LOG.APPEND
        )

    def train(self):
        # Set a signal handler.
        self._graceful_exit()

        performance_thresh = 0
        # step_flag = False

        q_validation_episodes = []
        score_validation_episodes = []
        for i_episode in range(self.init_episode, self.num_episodes):
            start_time = time.time()
            self.curr_episode = i_episode

            # Initialize the environment and state.
            self.env.reset()
            self.agent.state = self.env.get_state().to(self.device)

            self._train_episode()

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.cfg.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(
                    self.agent.policy_net.state_dict())

            if (
                    i_episode % self.cfg.VALIDATION.Q_VALIDATION_FREQUENCY == 0) and \
                    (self.cfg.VALIDATION.Q_VALIDATION_FREQUENCY != -1):
                validation_score = self.validate(val_type='q_value')
                self.q_validation_scores.append(validation_score)
                q_validation_episodes.append(i_episode)
                if self.cfg.VISUALIZE:
                    visualization.plot_validation_score(
                        self.q_validation_scores,
                        q_validation_episodes,
                        fig_num=3, y_label='Q value')

            if (
                    i_episode % self.cfg.VALIDATION.SCORE_VALIDATION_FREQUENCY == 0) and (
                    i_episode > 0) and \
                    (self.cfg.VALIDATION.SCORE_VALIDATION_FREQUENCY != -1):
                validation_score = self.validate(val_type='score')
                self.score_validation_scores.append(validation_score)
                score_validation_episodes.append(i_episode)
                if self.cfg.VISUALIZE:
                    visualization.plot_validation_score(
                        self.score_validation_scores,
                        score_validation_episodes,
                        fig_num=4, y_label='Duration')

            if i_episode % self.cfg.CKPT_SAVE_FREQ == 0:
                self._save_ckpt(i_episode)

            if self.cfg.SCHEDULER.SUCCESS_CRITERIA == "all_above_thresh":
                if all(list(map(lambda x : x > performance_thresh + self.cfg.SCHEDULER.PERFORMANCE_LEAP,
                            self.episode_durations[-self.cfg.SCHEDULER.EPISODES_SUCCESS_SEQUENCE:])))\
                        and self.episodes_from_scheduler_step >= self.cfg.SCHEDULER.MIN_EPISODES_BETWEEN_STEPS:

            # if (np.mean(self.episode_durations[-self.cfg.SCHEDULER.EPISODES_SUCCESS_SEQUENCE:]) \
            #         > performance_thresh + self.cfg.SCHEDULER.PERFORMANCE_LEAP) and (self.episode_durations[-1] > performance_thresh + self.cfg.SCHEDULER.PERFORMANCE_LEAP):
            #
                    performance_thresh += self.cfg.SCHEDULER.PERFORMANCE_LEAP
                    self.scheduler.step()
                    # self.scheduler_steps += 1
                    self.episodes_from_scheduler_step = 0
                else:
                    self.episodes_from_scheduler_step += 1


            # if (not step_flag) and np.mean(self.episode_durations[-10:]) > 200.0:
            #     step_flag = True
            #     self.scheduler.step()


            # elif (np.mean(self.episode_durations[-self.cfg.SCHEDULER.EPISODES_SUCCESS_SEQUENCE:]) \
            #         < performance_thresh - self.cfg.SCHEDULER.PERFORMANCE_LEAP) and (self.episode_durations[-1] < performance_thresh - self.cfg.SCHEDULER.PERFORMANCE_LEAP):
            #
            #     performance_thresh = max(performance_thresh - self.cfg.SCHEDULER.PERFORMANCE_LEAP, 0)
            #     self.scheduler.
            #     self.scheduler_steps = max(self.scheduler_steps - 1, 0)

            # Scalar logging.
            self.logger.log_tabular('Epoch', i_episode)
            self.logger.log_tabular('TotalGradientSteps', self.steps_done)
            self.logger.log_tabular('EpsilonThreshold', self.eps_threshold)
            self.logger.log_tabular('EpisodeDuration',
                                    self.episode_durations[-1])
            self.logger.log_tabular('MeanEpisodeDuration',
                                    np.mean(self.episode_durations[-100:]))

            if len(self.q_validation_scores) != 0:
                self.logger.log_tabular('QValidation',
                                        self.q_validation_scores[-1])
            else:
                self.logger.log_tabular('QValidation', -1)
            if len(self.score_validation_scores) != 0:
                self.logger.log_tabular('ScoreValidation',
                                        self.score_validation_scores[-1])
            else:
                self.logger.log_tabular('ScoreValidation', -1)

            self.logger.log_tabular('LR', self.optimizer.state_dict()['param_groups'][0]['lr'])
            self.logger.log_tabular('Loss', self.episode_mean_losses[-1])
            self.logger.log_tabular('Time', time.time() - start_time)

            self.metric = self.logger.log_current_row['MeanEpisodeDuration']
            self.logger.dump_tabular()


    def _train_episode(self):
        """

        Returns:

        """
        state_history = []
        losses = []

        for t in count():
            # Select and perform an action
            self.eps_threshold = (self.cfg.EPS_END + (
                        self.cfg.EPS_START - self.cfg.EPS_END) \
                                 * math.exp(
                -1. * self.steps_done / self.cfg.EPS_DECAY)) * (math.sqrt(self.cfg.SCHEDULER.GAMMA) ** self.scheduler_steps)

            action = self.agent.select_action(self.eps_threshold)
            self.steps_done += 1
            _, reward, done, _ = self.env.step(action.item())

            reward = torch.tensor([reward], device=self.device)

            # Observe new state
            next_state = None
            if not done:
                next_state = self.env.get_state().to(self.device)

            # Store the transition in memory
            self.memory.push(self.agent.state, action, next_state, reward)

            # Move to the next state
            self.agent.state = next_state

            # Perform one step of the optimization (on the target network)
            step_loss = self.step()
            if step_loss is not None:
                losses.append(step_loss.item())

            if done:
                mean_loss = -1
                if len(losses) != 0:
                    mean_loss = np.array(losses).mean()
                self.episode_mean_losses.append(mean_loss)

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
                                                batch.next_state)),
                                      device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net.
        state_action_values = self.agent.policy_net(state_batch).gather(1,
                                                                        action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.

        # self.cfg.OPT_LEVEL == "O0":
        data_type = torch.float
        # else:
        #     data_type = torch.half

        next_state_values = torch.zeros(self.cfg.BATCH_SIZE,
                                        device=self.device,
                                        dtype=data_type)

        next_state_values[non_final_mask] = \
            self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.cfg.GAMMA)\
                                       + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()

        # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()

        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def validate(self, val_type='q_value'):
        with torch.no_grad():
            validation_values = []
            if val_type == 'q_value':
                for state in self.env_random_states:
                    current_state_q = max(
                            self.policy_net(
                                    state.to(self.device)).data.cpu().numpy()[0])
                    validation_values.append(current_state_q)
                validation_value = np.mean(validation_values)
            elif val_type == 'score':
                for state in self.env_initial_states_screens:
                    # Initialize the state.
                    self.env.reset()
                    self.agent.state = state
                    _, episode_duration = self.agent._play_episode()
                    validation_values.append(episode_duration)
                validation_value = np.mean(validation_values)
        return validation_value

    def _save_ckpt(self, episode=None):
        checkpoint = {
            'model': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'amp': amp.state_dict(),
            'steps_done': self.steps_done,
            'init_episode': self.curr_episode
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
            # amp.load_state_dict(checkpoint['amp'])
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

        if not os.path.isdir(self.cfg.CKPT_SAVE_DIR):
            os.makedirs(self.cfg.CKPT_SAVE_DIR)
            print(f'Created results dir at {self.cfg.CKPT_SAVE_DIR}')

        files = os.listdir(self.cfg.CKPT_SAVE_DIR)
        newest_ckpt_name = None
        if files != []:
            newest_ckpt_name = max(files, key=lambda x: int(
                x.split('.')[0].split('_')[-1]))
        return newest_ckpt_name


class DQNAgent(object):
    def __init__(self, policy_net, n_actions, device, env, epsilon=0.0):
        self.policy_net = policy_net
        self.state = None
        self.n_actions = n_actions
        self.device = device
        self.env = env
        self.epsilon = epsilon

    def _play_episode(self):
        """

        Returns:

        """
        states = []
        for t in count():
            states.append(self.state)

            # Select and perform an action
            action = self.select_action(eps_threshold=0)
            # self.steps_done += 1
            _, _, done, _ = self.env.step(action.item())

            if not done:
                self.state = self.env.get_state().to(self.device)

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
