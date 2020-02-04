from yacs.config import CfgNode as CN


def get_default_params():
    default_params = dict()

    default_params['PATHS'] = {
        'ESTIMATED_VALIDATION_SET_PATH': '',
        'EMPIRICAL_VALIDATION_SET_PATH': '',
    }

    default_params['TRAIN'] = {
        'BATCH_SIZE': 128,
        'GAMMA': 0.999,
        'TARGET_UPDATE': 10,
        'EPS_START': 0.9,
        'EPS_END':  0.05,
        'EPS_DECAY': 200,
        'REPLAY_MEMORY_SIZE': 10000,

        'NUM_EPISODES': 50000,
        'OPT_LEVEL': 'O1',

        'ESTIMATED_VALIDATION_FREQUENCY': 10,
        'EMPIRICAL_VALIDATION_FREQUENCY': 100,
        'CKPT_SAVE_FREQ': 10,
        'CKPT_PATH': '',
        'CKPT_SAVE_DIR': '',

        'LOG': {
            'OUTPUT_DIR': '',
            'OUTPUT_FNAME': 'progress.txt',
            'EXP_NAME': '',
            'SAVE_FREQ': 1,
        },

        'VISUALIZE': False,
    }

    return default_params


cfg = CN(get_default_params())
