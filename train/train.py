import argparse

import gym
import torch
import torch.optim as optim
from sacred import Experiment
from sacred.observers import FileStorageObserver

import algorithms.dqn.trainer
import utils.yaml_utils
from algorithms.dqn.model import dqn_vanilla
from algorithms.dqn.utils import replay_mem
from config.default_config import cfg
from utils import gym_utils

# from apex import amp, optimizers

ex = Experiment()


@ex.main
def main():
    # study = optuna.create_study(
    #     sampler=optuna.samplers.TPESampler(),
    #     direction='maximize',
    #     study_name='cartpole_rl',
    #     storage=f'sqlite:///cartpole_rl_hp_opt_1.db',
    #     load_if_exists=True
    # )
    #
    # study.optimize(train, n_trials=100)
    train()


def train(trial=None):
    if trial:
        cfg.TRAIN.GAMMA = trial.suggest_uniform('GAMMA', 0.95, 0.99)
        cfg.TRAIN.EPS_DECAY = trial.suggest_uniform('EPS_DECAY', 100, 600)
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)


    env = gym_utils.EnvWrapper(gym.make('CartPole-v0').unwrapped, num_frames=4)
    env.reset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Running on CPU!!!")
    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_state = env.get_state().to(device)
    _, _, screen_height, screen_width = init_state.shape
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    policy_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(
        device)
    target_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions).to(
        device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    # optimizer = optim.RMSprop(policy_net.parameters())
    optimizer = optim.Adam(policy_net.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5,
                                                       last_epoch=-1)
    # if torch.cuda.is_available():
    #     [target_net, policy_net], optimizer = amp.initialize(
    #         [target_net, policy_net], optimizer,
    #         opt_level=cfg.TRAIN.OPT_LEVEL)

    memory = replay_mem.ReplayMemory(cfg.TRAIN.REPLAY_MEMORY_SIZE)
    # env_random_states = pickle.load(
    #     open(cfg.PATHS.Q_VALIDATION_SET_PATH, 'rb'))
    # env_initial_states_screens = pickle.load(
    #     open(cfg.PATHS.SCORE_VALIDATION_SET_PATH, 'rb'))
    # env_initial_states_screens = env_initial_states_screens[
    #                              :cfg.TRAIN.VALIDATION.SCORE_VALIDATION_SIZE]
    agent = algorithms.dqn.trainer.DQNAgent(policy_net, n_actions, device, env,
                                            cfg.TRAIN.EPS_END)
    trainer = algorithms.dqn.trainer.DQNTrainer(
        cfg.TRAIN, env, agent, target_net, policy_net, memory, optimizer,
        cfg.TRAIN.NUM_EPISODES, device,
        scheduler
    )
    trainer.train()
    print('Complete')

    return trainer.metric


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
