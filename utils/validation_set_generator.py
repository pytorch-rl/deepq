
import random
import pickle
import torch
import torch.optim as optim
import gym

from sacred import Experiment
from sacred.observers import FileStorageObserver

from algorithms.dqn.model import dqn_vanilla
from algorithms.dqn.utils import replay_mem
import algorithms.dqn.trainer

from train.train import parse_args

import utils.yaml_utils
from utils import gym_utils

from config.default_config import cfg


ex = Experiment()


VALIDATION_SET_SIZE = 1000
EPISODES_TO_PLAY = 1000
SEED = 7
PATH_TO_SAVE = '/raid/algo/SOCISP_SLOW/ADAS/Courses/mspacman/data/validation_set_random_' + str(VALIDATION_SET_SIZE) + '.pickle'


# @ex.main
# def generate_initial_states_list():
#     state_list = []
#     env = gym.make('CartPole-v0').unwrapped
#     env.seed(7)
#     for i in range(VALIDATION_SET_SIZE):
#         env.reset()
#         state_list.append(gym_utils.get_screen(env))
#     pickle.dump(state_list, open('validation_set_initial.pickle', 'wb'))


@ex.main
def generate_random_states_list():
    """
    generates a validation set of randomly chosen samples from episodes played with randomly
    initialized policy net and saves it as .pickle
    """
    env = gym.make('CartPole-v0').unwrapped
    env.seed(SEED)
    env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_actions = env.action_space.n
    init_screen = gym_utils.get_screen(env).to(device)
    _, _, screen_height, screen_width = init_screen.shape
    policy_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(device)
    target_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(device)
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = replay_mem.ReplayMemory(10000)
    num_episodes = 0
    env_state_list = []

    agent = algorithms.dqn.trainer.DQNAgent(policy_net, n_actions, device)
    trainer = algorithms.dqn.trainer.DQNTrainer(cfg.TRAIN, env, agent, target_net,
                                                policy_net, memory, optimizer, num_episodes,
                                                device, env_state_list)

    state_list = []
    for i_episode in range(EPISODES_TO_PLAY):
        # Initialize the environment and state
        env.reset()
        last_screen = gym_utils.get_screen(env).to(device)
        current_screen = gym_utils.get_screen(env).to(device)
        trainer.agent.state = current_screen - last_screen
        state_list += trainer._play_episode(current_screen, return_state_history=True)

    random.seed(SEED)
    validation_set = random.sample(state_list, k=VALIDATION_SET_SIZE)
    pickle.dump(validation_set, open(PATH_TO_SAVE, 'wb'))


if __name__ == '__main__':
    args = parse_args()

    if args.cfg_path != '':
        utils.yaml_utils.load_from_yaml(args.cfg_path, cfg)
        ex.add_config(cfg)

    if args.file_storage_path != '':
        ex.observers.append(FileStorageObserver(args.file_storage_path))

    ex.run()
