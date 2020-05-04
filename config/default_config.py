from yacs.config import CfgNode as CN


def get_default_params():
    default_params = dict()

    default_params['PATHS'] = {
        'Q_VALIDATION_SET_PATH': '', # Path to random initial states set for Q validation
        'SCORE_VALIDATION_SET_PATH': '', # Path to ranom states set for score validation
    }

    default_params['TRAIN'] = {
        'BATCH_SIZE': 128,
        'GAMMA': 0.999,  # Reward discount factor per step
        'TARGET_UPDATE': 1000,  # Episodes between updates of target net
        'EPS_START': 0.9,  # Initial value of epsilon (exploration probability)
        'EPS_END':  0.05,  # Final value of epsilon
        'EPS_DECAY': 200,  # Epsilon decay rate
        'REPLAY_MEMORY_SIZE': 10000,  # Max number of states in replay memory
        'LEARNING_RATE': 5e-4,

        'SCHEDULER': {
            'GAMMA': 0.1,
            'INITIAL_PERFORMANCE_THRSH': 100,  # Number of steps per episode from which to reducing learning rate
            'PERFORMANCE_LEAP': 50,  # Improvement in number of steps per episode required for reducing learning rate
            'EPISODES_SUCCESS_SEQUENCE': 5,  # Number of successful episodes in a row required for reducing learning rate
            'COOLDOWN': 10,  # Minimum number of episodes between scheduler steps
        },

        'NUM_EPISODES': 50000,  # Total number of episodes to train
        'CKPT_SAVE_FREQ': 10,  # Number of episodes between checkpoint save
        'CKPT_PATH': '',  # Path to load checkpoint
        'CKPT_SAVE_DIR': '',  # Path to save checkpoint

        'LOG': {
            'OUTPUT_DIR': '', # Path to save log
            'OUTPUT_FNAME': 'progress.txt',
            'EXP_NAME': '',
            'APPEND': True
        },

        'VALIDATION': {
            'Q_VALIDATION_FREQUENCY': 10,  # Number of episodes between Q validation execution
            'SCORE_VALIDATION_FREQUENCY': 100,  # Number of episodes between score validation execution
            'SCORE_VALIDATION_SIZE': 10,  # Number of states for score validation
        },

        'VISUALIZE': False,
    }

    return default_params


cfg = CN(get_default_params())
