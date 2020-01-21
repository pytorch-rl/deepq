import random
import torch
import math


EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


def select_action(state, steps_done, policy_net, n_actions, device):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1), steps_done
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done
