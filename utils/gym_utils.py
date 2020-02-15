import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(env, screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(env, screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW).
    return resize(screen).unsqueeze(0).mean(axis=1, keepdims=True)


class EnvWrapper:
    def __init__(self, env, num_frames):
        self.num_frames = num_frames
        self._frames = []
        self.env = env
        self.action_space = self.env.action_space
        self.seed = self.env.seed

    def reset(self):

        self.env.reset()

        self._frames = []
        self._frames.append(get_screen(self.env))
        for env_step in range(self.num_frames - 1):
            self.env.step(0)
            self._frames.append(get_screen(self.env))

    def step(self, action):
        observation, reward, done, lives = self.env.step(action)
        cur_screen = get_screen(self.env)

        del self._frames[0]
        self._frames.append(cur_screen)

        return observation, reward, done, lives

    def get_state(self):
        return torch.stack(self._frames).squeeze(1).permute(1,0,2,3)


#
# if __name__ == '__main__':
#
#     import gym
#     env = gym.make('CartPole-v0').unwrapped
#     env.reset()
#     ew = EnvWrapper(env, 4)
#
#     ew.reset_state()
#     x = ew.get_state()


