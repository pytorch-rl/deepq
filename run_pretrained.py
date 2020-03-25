import argparse

import gym
import matplotlib.pyplot as plt
import torch
from matplotlib import animation

import algorithms
import algorithms.dqn.trainer
from algorithms.dqn.model import dqn_vanilla
from utils import gym_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Inference config.')

    parser.add_argument('--cfg_path', type=str, required=False, default='',
                        help='Path to YAML config file.')
    parser.add_argument('--results_dir', type=str, required=False,
                        default='./results',
                        help='FileStorageObserver path.')

    return parser.parse_args()


def run_pretrained():
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
    policy_net = dqn_vanilla.DQN(screen_height, screen_width, n_actions)
    policy_net.to(device).eval()

    agent = algorithms.dqn.trainer.DQNAgent(policy_net, n_actions, device, env)
    ckpt_path = './assets/checkpoint.pt'
    runner = CartpoleRunner(agent, env, policy_net, device, ckpt_path)
    runner.run()
    env.close()
    print('\n Session ended')
    return None


class CartpoleRunner(object):
    def __init__(self, agent, env, policy_net, device, ckpt_path):
        self.agent = agent
        self.env = env
        self.policy_net = policy_net
        self.device = device
        self.steps_done = 0
        self.ckpt_path = ckpt_path
        self.init_episode = 0

        self._load_ckpt()

    def run(self):

        frames = []
        self.agent.state = self.env.get_state().to(self.device)

        while True:

            action = self.agent.select_action(eps_threshold=0)
            self.steps_done += 1
            _, reward, done, _ = self.env.step(action.item())

            frames.append(self.env.render(mode="rgb_array"))

            # Observe new state
            if not done:
                next_state = self.env.get_state().to(self.device)
                self.agent.state = next_state

            if done:
                # save_frames_as_gif(frames)
                return None
            # Move to the next state

    def _load_ckpt(self):

        print('Loading {}'.format(self.ckpt_path))
        checkpoint = torch.load(self.ckpt_path)
        self.policy_net.load_state_dict(checkpoint['model'])


def save_frames_as_gif(frames, path='./', filename='results/gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


if __name__ == '__main__':
    args = parse_args()

    # if args.cfg_path != '':
    #     utils.yaml_utils.load_from_yaml(args.cfg_path, cfg)

    run_pretrained()
