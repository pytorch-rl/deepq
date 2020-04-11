import argparse

import pickle
import torch
import gym

import utils.yaml_utils
import utils.gym_utils

from algorithms.dqn.model import dqn_vanilla
from algorithms.dqn.utils import replay_mem
import algorithms.dqn.trainer
import algorithms.dqn.agent
from config.default_config import cfg


def main():
    """configuration and training"""
    args = parse_args()
    if args.cfg_path != '':
        utils.yaml_utils.load_from_yaml(args.cfg_path, cfg)

    train()


def train():
    """Initialization and training of a cart-pole DQN agent"""

    # initialization
    env = utils.gym_utils.EnvWrapper(gym.make('CartPole-v0').unwrapped, num_frames=4)
    env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        print("Running on CPU!!!")

    init_state = env.get_state().to(device)
    _, _, screen_height, screen_width = init_state.shape
    # Get number of possible actions from gym action space
    n_actions = env.action_space.n
    policy_net = dqn_vanilla.DQN(screen_height,
                                 screen_width,
                                 n_actions).to(device)
    target_net = dqn_vanilla.DQN(screen_height,
                                 screen_width,
                                 n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(),
                                 lr=cfg.TRAIN.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=cfg.TRAIN.SCHEDULER.GAMMA,
                                                       last_epoch=-1)
    memory = replay_mem.ReplayMemory(cfg.TRAIN.REPLAY_MEMORY_SIZE)

    # load validation sets
    to_device = lambda x: x.to(device)
    if cfg.TRAIN.VALIDATION.Q_VALIDATION_FREQUENCY != -1:
        env_random_states = \
            pickle.load(open(cfg.PATHS.Q_VALIDATION_SET_PATH, 'rb'))
        env_random_states = list(map(to_device, env_random_states))
    else:
        env_random_states = []

    if cfg.TRAIN.VALIDATION.SCORE_VALIDATION_FREQUENCY != -1:
        env_initial_states = pickle.load(
            open(cfg.PATHS.SCORE_VALIDATION_SET_PATH, 'rb'))
        env_initial_states = env_initial_states[
                                     :cfg.TRAIN.VALIDATION.SCORE_VALIDATION_SIZE]
        env_initial_states = list(map(to_device, env_initial_states))
    else:
        env_initial_states = []

    agent = algorithms.dqn.agent.DQNAgent(policy_net, n_actions, device, env,
                                            cfg.TRAIN.EPS_END)
    trainer = \
        algorithms.dqn.trainer.DQNTrainer(
            cfg.TRAIN, env, agent, target_net, memory, optimizer,
            cfg.TRAIN.NUM_EPISODES, device,
            scheduler, env_random_states, env_initial_states
    )

    trainer.train()
    print('Complete')

def parse_args():
    """Commandline argument parser for configuration

    Returns: Configuraion object
    """
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
    main()

