import os
import random
import pickle

import torch
import gym

from algorithms.dqn.model import dqn_vanilla
from algorithms.dqn.utils import replay_mem
import algorithms.dqn.trainer

from train.train import parse_args

import utils.yaml_utils
from utils import gym_utils

from config.default_config import cfg


VALIDATION_SET_SIZE = 1000
EPISODES_TO_PLAY = 100
SEED = 7
PATH_TO_SAVE = '/raid/algo/SOCISP_SLOW/ADAS/Courses/mspacman/data/'

GENERATE_INIT_STATES = True
GENERATE_RANDOM_STATES = True


def generate_init_states():
    """
    Generates a validation random initial states and screens and saves it as .pickle.
    """
    env = gym_utils.EnvWrapper(gym.make('CartPole-v0').unwrapped, num_frames=4)
    env.seed(SEED)
    env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_states = []

    for i_episode in range(EPISODES_TO_PLAY):
        init_state = env.get_state().to(device)
        env.reset()
        init_states.append(init_state)

    file_path = os.path.join(PATH_TO_SAVE, 'validation_set_initial_' + str(VALIDATION_SET_SIZE) + '.pickle')
    pickle.dump(init_states, open(file_path, 'wb'))


def generate_random_states():
    """
    Generates a validation set of randomly chosen samples from episodes played with randomly
    initialized policy net and saves it as .pickle.
    """
    # env = gym.make('CartPole-v0').unwrapped
    env = gym_utils.EnvWrapper(gym.make('CartPole-v0').unwrapped, num_frames=4)
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

    agent = algorithms.dqn.trainer.DQNAgent(policy_net, n_actions, device, env)
    trainer = algorithms.dqn.trainer.DQNTrainer(cfg.TRAIN, env, agent, target_net,
                                                policy_net, memory, optimizer, num_episodes,
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
    file_path = os.path.join(PATH_TO_SAVE, 'validation_set_random_' + str(VALIDATION_SET_SIZE) + '.pickle')
    pickle.dump(validation_set, open(file_path, 'wb'))


if __name__ == '__main__':

    args = parse_args()

    if args.cfg_path != '':
        utils.yaml_utils.load_from_yaml(args.cfg_path, cfg)

    if GENERATE_INIT_STATES:
        generate_init_states()
    if GENERATE_RANDOM_STATES:
        generate_random_states()
