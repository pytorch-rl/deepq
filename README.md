# deepq

## Project Summary

### Introduction

- What our goal were - Solve cartpole using raw input
- Was supposed to be a toy example on the way and turned out differently
- Based on the original pytorch tutorial (we were not able to make it work consistently).
- Found to be unstable and inconsistent.
- Did not find any solution with a CNN (we found failed attempts)
- Added 2 dense layers to the model
- Tried hyperparameter optimization for no avail 
- Was able to win the game using a scheduler

### Results

The good news: It learns!

![](assets/cartpole_example.gif)

The bad news: It does so very inconsistently.

![](assets/multi_trial_analysis.png)

### Conclusions

- RL is harder than supervised learning
- 

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
