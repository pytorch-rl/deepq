import os
import argparse
import random
import pickle

import torch
import gym

from algorithms.dqn.model import dqn_vanilla
from algorithms.dqn.utils import replay_mem
import algorithms.dqn.trainer
import algorithms.dqn.agent

from train.train import parse_args

import utils.gym_utils
import utils.yaml_utils

from config.default_config import cfg


VALIDATION_SET_SIZE = 1000
EPISODES_TO_PLAY = 100
SEED = 7


def generate_init_states(path_to_save):
    """
    Generates a validation random initial states and screens and saves it as .pickle.

    Args:
        path_to_save: path to save validtion set
    """
    env = utils.gym_utils.EnvWrapper(gym.make('CartPole-v0').unwrapped, num_frames=4)
    env.seed(SEED)
    env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_states = []

    for i_episode in range(EPISODES_TO_PLAY):
        init_state = env.get_state().to(device)
        env.reset()
        init_states.append(init_state)

    file_path = os.path.join(path_to_save, 'validation_set_initial_' + str(VALIDATION_SET_SIZE) + '.pickle')
    pickle.dump(init_states, open(file_path, 'wb'))


def generate_random_states(path_to_save):
    """
    Generates a validation set of randomly chosen samples from episodes played with randomly
    initialized policy net and saves it as .pickle.

    Args:
        path_to_save: path to save validtion set
    """
    env = utils.gym_utils.EnvWrapper(gym.make('CartPole-v0').unwrapped, num_frames=4)
    env.seed(SEED)
    env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_actions = env.action_space.n

    init_state = env.get_state().to(device)
    _, _, screen_height, screen_width = init_state.shape
    policy_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(device)
    target_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters())
    memory = replay_mem.ReplayMemory(100)
    num_episodes = 0
    env_random_states = []
    env_initial_states = []
    scheduler = None

    agent = algorithms.dqn.agent.DQNAgent(policy_net, n_actions, device, env)
    trainer = algorithms.dqn.trainer.DQNTrainer(cfg.TRAIN, env, agent, target_net,
                                                memory, optimizer, num_episodes,
                                                device, scheduler, env_random_states, env_initial_states)

    state_list = []
    for i_episode in range(EPISODES_TO_PLAY):
        env.reset()
        init_state = env.get_state().to(device)
        trainer.agent.state = init_state
        episode_states, _ = trainer.agent._play_episode()
        state_list += episode_states

    random.seed(SEED)
    validation_set = random.sample(state_list, k=VALIDATION_SET_SIZE)
    file_path = os.path.join(path_to_save, 'validation_set_random_' + str(VALIDATION_SET_SIZE) + '.pickle')
    pickle.dump(validation_set, open(file_path, 'wb'))


def parse_args():
    """Commandline argument parser for configuration

    Returns: Configuraion object
    """
    parser = argparse.ArgumentParser(description='Validation set generator.')

    parser.add_argument('--files_storage_path',
                        type=str,
                        required=True,
                        default='~/algorithms/dqn/validation/validation_set/',
                        help='Validation set path.')
    parser.add_argument('--cfg_path',
                        type=str,
                        required=False,
                        default='', # todo(amitka) - add default path in repo
                        help='Path to YAML config file generating for random state validation set.')


    return parser.parse_args()


def main():
    """configuration validation set generation"""

    args = parse_args()
    path_to_save = args.files_storage_path
    if args.cfg_path != '':
        utils.yaml_utils.load_from_yaml(args.cfg_path, cfg)

    generate_init_states(path_to_save)
    generate_random_states(path_to_save)


if __name__ == '__main__':
    main()
