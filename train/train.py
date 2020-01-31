import argparse

import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F

import pickle

from config.default_config import cfg
from apex import amp, optimizers

import utils.yaml_utils
from algorithms.dqn.utils import replay_mem, dqn_algo
from algorithms.dqn.model import dqn_vanilla
from utils import gym_utils

import algorithms.dqn.trainer

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()


@ex.main
def main():
    env = gym.make('CartPole-v0').unwrapped

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Running on CPU!!!")

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    env.reset()

    init_screen = gym_utils.get_screen(env).to(device)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(device)
    target_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(params=policy_net.parameters(), lr=cfg.TRAIN.LEARNING_RATE, alpha=cfg.TRAIN.ALPHA)

    [target_net, policy_net], optimizer = amp.initialize([target_net, policy_net],
                                          optimizer, opt_level=cfg.TRAIN.OPT_LEVEL)

    memory = replay_mem.ReplayMemory(cfg.TRAIN.REPLAY_BUFFER_SIZE)

    env_state_list = pickle.load(open(cfg.PATHS.VALIDATION_SET_PATH, 'rb'))

    agent = algorithms.dqn.trainer.DQNAgent(policy_net, n_actions, device)
    trainer = algorithms.dqn.trainer.DQNTrainer(
        cfg.TRAIN, env, agent, target_net, policy_net, memory, optimizer,
        device, env_state_list)

    trainer.train()

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
