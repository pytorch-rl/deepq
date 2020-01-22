import argparse

import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F

from config.default_config import cfg

import utils.yaml_utils
from algorithms.dqn.utils import replay_mem, dqn_algo
from algorithms.dqn.model import dqn_vanilla
from utils import gym_utils
from utils import visualization

import algorithms.dqn.trainer

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()


def optimize_model(memory, policy_net, target_net, optimizer, device):
    if len(memory) < cfg.TRAIN.BATCH_SIZE:
        return

    transitions = memory.sample(cfg.TRAIN.BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = replay_mem.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device,
                                  dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(cfg.TRAIN.BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = \
        target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * cfg.TRAIN.GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


@ex.main
def main():
    env = gym.make('CartPole-v0').unwrapped

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    env.reset()
    init_screen = gym_utils.get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(
        device)
    target_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(
        device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = replay_mem.ReplayMemory(10000)

    episode_durations = []

    steps_done = 0

    num_episodes = 50

    agent = algorithms.dqn.trainer.DQNAgent(policy_net, n_actions, device)
    trainer = algorithms.dqn.trainer.DQNTrainer()

    trainer.train()



    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = gym_utils.get_screen(env, device)
        current_screen = gym_utils.get_screen(env, device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action, steps_done = dqn_algo.select_action(state, steps_done,
                                                        policy_net, n_actions, device)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = gym_utils.get_screen(env, device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(memory, policy_net, target_net, optimizer, device)
            if done:
                episode_durations.append(t + 1)
                visualization.plot_durations(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % cfg.TRAIN.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Inference config.')

    parser.add_argument('--cfg_path',
                        type=str,
                        required=False,
                        default='',
                        help='Path to YAML config file.')
    parser.add_argument('--file_storage_path',
                        type=str,
                        required=False,
                        default='',
                        help='FileStorageObserver path.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.cfg_path != '':
        utils.yaml_utils.load_from_yaml(args.cfg_path, cfg)
        ex.add_config(cfg)

    if args.file_storage_path != '':
        ex.observers.append(FileStorageObserver(args.file_storage_path))

    ex.run()
