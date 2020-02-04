
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
EPISODES_TO_PLAY = 10
SEED = 7
PATH_TO_SAVE = '/raid/algo/SOCISP_SLOW/ADAS/Courses/mspacman/data/validation_set_initial_' + str(VALIDATION_SET_SIZE) + '.pickle'


@ex.main
def generate_init_states_list():
    """
    generates a validation random initial states and screens and saves it as .pickle
    """
    env = gym.make('CartPole-v0').unwrapped
    env.seed(SEED)
    env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_state_list = []
    init_screen_list = []
    for i_episode in range(EPISODES_TO_PLAY):
        env.reset()
        last_screen = gym_utils.get_screen(env).to(device)
        current_screen = gym_utils.get_screen(env).to(device)
        init_state = current_screen - last_screen
        init_state_list.append(init_state)
        init_screen_list.append(current_screen)
    init_state_screen_list = list(zip(init_state_list, init_screen_list))
    # pickle.dump(init_state_list, open(PATH_TO_SAVE, 'wb'))
    pickle.dump(init_state_screen_list, open(PATH_TO_SAVE, 'wb'))


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
    env_random_state_list = []
    env_initial_state_list = []


    agent = algorithms.dqn.trainer.DQNAgent(policy_net, n_actions, device, env)
    trainer = algorithms.dqn.trainer.DQNTrainer(cfg.TRAIN, env, agent, target_net,
                                                policy_net, memory, optimizer, num_episodes,
                                                device, env_random_state_list, env_initial_state_list)

    state_list = []
    for i_episode in range(EPISODES_TO_PLAY):
        # Initialize the environment and state
        env.reset()
        last_screen = gym_utils.get_screen(env).to(device)
        current_screen = gym_utils.get_screen(env).to(device)
        trainer.agent.state = current_screen - last_screen
        states, _ = trainer.agent._play_episode(current_screen)
        state_list += states

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
