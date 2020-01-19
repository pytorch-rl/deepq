
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from dqn_objects import replay_mem
from model import dqn_vanilla
import utils.gym_utils

env = gym.make('CartPole-v0') # .unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    env.reset()
    plt.figure()


    plt.imshow(utils.gym_utils.get_screen(env, device).cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


if __name__ == '__main__':
    main()
