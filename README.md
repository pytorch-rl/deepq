# deepq

## Project Summary

### Introduction

This repository is a results of a final course project for TAUs *Deep Learning Course* by Raja Giryes.

Our goal was a first dive to the world of RL by implementing DQN using a DNN policy network to win an Atari game using raw inputs (screen pixels as opposed to some other state).

The Cartpol game emulated by the OpenAI gym was chosen as an initial problem since we could not find any implementation successfully winning Cartpole[^1] via DQN **and CNN** online aroused our interest.   

We based our work on the official [pytorch DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) that indeed tries to win Cartpol with DQN+CNN but could not win the game under the gym definitions. 

Coming from the world of supervised learning, we were baffled by how **unstable** and **inconsistent** training a DQN agent can be, these problems prevented from the agent from winning the game. 

One must note that all of the DQN Cartpole solutions running around online solve the problem using a 4 dimensional input ($x$, $v_x$, $\theta$, $v_\theta$) making this a **much** simpler problem than the one reading the screen.   

Most of our effort trying to to improve these properties, we experimented with architectural changes to the policy network, an extensive hyperparameter search and learning rate scheduling methods. These, only improved stability and consistency mildly.

Eventually, we were able to bypass stability issues and win the game using a harsh learning rate regime in which the scheduler was score aware and reduced the learning rate drastically when performance was good enough to win the game.

[^1]: From https://gym.openai.com/envs/CartPole-v0/: "*CartPole-v0 defines "solving" as getting average reward of 195.0 over 100 consecutive trials.*"

### Results

#### The good new

> We were able to consistently learn to play (much) better than random!

Below is one game played by one of our trained agents, it is clear that the agent learned the rules and it able to play the game well. 

|  ![](assets/cartpole_example.gif)    |
| ---- |
| One of our winning agents playing the game |



#### The bad new

> We were not able to win the game consistently

Below is an analysis of 90 training sessions with identical parameters of our final configuration, each line represents the running average of the last 100 episode durations training. 

This figure shows that:

1. The agent consistently learns to play the game better than random.
2. The learning process is extremely unstable, "forgetting" episodes are frequent.
3. Some sessions were able to play the game (all the lines that pass the 200 line)

|    ![](assets/multi_trial_analysis.png)  |
| ---- |
|      |

### Conclusions

- RL is harder than supervised learning

## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN 10

## Getting Started
### Installation

- Clone this repository:

```bash
git clone https://github.com/pytorch-rl/mspacman.git
```

- Add repository path to your PYTHONPATH:

```bash
cd deepq
export PYTHONPATH=$PWD:PYTHONPATH
```

- Install requirements:

```bash
pip install -r requirements.txt`
```

### Training

- We assume that the pwd is ``deepq``. We will use a ``results`` directory in 
the repository for the follweing examples.

- The training code performs online validation. So first, we need to create
a validation set using:

```bash
python validation/validation_set_generator.py
```

Note that you can disabel validation using:

```python
 Q_VALIDATION_FREQUENCY: -1
 SCORE_VALIDATION_FREQUENCY: -1
```

- Now you are ready to train your agent:

```bash
python train/train.py --cfg_path="./config/best_train_cfg.yaml"
```

#### Hyperparameters

#### Tips

### Apply a pretrained model

Run `run_pretrained.py`:
```bash
python run_pretrained.py
```

With the possibility of defining your own trained checkpoint `--ckpt_path` and disabling gif saving via entering 'False' in  `--save_gif`. Note that gif saving is dependent on having Imagemagick installed (sudo apt install imagemagick).

By default, the agent is loading a predefined model `assets/checkpoint.pt` and saves the `.gif` file to `results/cartpole_example.gif`.


## Citation

The algorithm used in this project is based on:

```
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. and Riedmiller, M., 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
```

## Acknowledgments

Our code is based upon the [official pytorch DQN tutorial](https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py).  
