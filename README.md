# deepq

![](assets/cartpole_example.gif)

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN 10

## Getting Started
### Installation

- Clone this repo:

```bash
git clone https://github.com/pytorch-rl/mspacman.git
```

- Add repo path to your PYTHONPATH:

```bash
cd deepq
export PYTHONPATH=$PWD:PYTHONPATH
```

- Type the command `pip install -r requirements.txt`.

### Training

- We assume the pwd is ``deepq``. We will use the ``resutls`` folder in 
the repository for the examples.

- The training code also performs online validation. So first, we need to create
a validation set using:

```bash
python validation/validation_set_generator.py
```

- Now you are ready to train your agent:

```bash
python train/train.py --cfg_path="../config/best_train_cfg.yaml"
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

Our code is inspired by the [official pytorch DQN tutorial](https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py).  
