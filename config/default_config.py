from yacs.config import CfgNode as CN


def get_default_params():
    default_params = dict()

    default_params['PATHS'] = {
        'VALIDATION_SET_PATH': ''
    }

    default_params['TRAIN'] = {
        'BATCH_SIZE': 128,
        'GAMMA': 0.999,
        'TARGET_UPDATE': 10000,
        'EPS_START': 0.9,
        'EPS_END':  0.05,
        'EPS_DECAY': 200,

        'LEARNING_RATE': 0.00025,
        'ALPHA': 0.250,
        'REPLAY_BUFFER_SIZE': 1000000,
        'LEARNING_STARTS': 50000,

        'NUM_EPISODES': 100000,

        'OPT_LEVEL': 'O1',

        'VALIDATE_FREQUENCY': 10,

        'CKPT_SAVE_FREQ': 100,
        'CKPT_PATH': '',
        'CKPT_SAVE_DIR': '',

        'LOG': {
            'OUTPUT_DIR': '',
            'OUTPUT_FNAME': 'progress.txt',
            'EXP_NAME': '',
            'SAVE_FREQ': 1,
            'APPEND': True
        },

        'VISUALIZE': False,
    }

    return default_params


cfg = CN(get_default_params())
