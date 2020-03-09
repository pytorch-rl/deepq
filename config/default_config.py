from yacs.config import CfgNode as CN


def get_default_params():
    default_params = dict()

    default_params['HP_OPTIM'] = {
		'NUM_TRIALS': 1,
	}
	
    default_params['PATHS'] = {
        'Q_VALIDATION_SET_PATH': '',
        'SCORE_VALIDATION_SET_PATH': '',
    }

    default_params['TRAIN'] = {
        'BATCH_SIZE': 128,
        'GAMMA': 0.999,
        'TARGET_UPDATE': 1000,
        'EPS_START': 0.9,
        'EPS_END':  0.05,
        'EPS_DECAY': 200,
        'REPLAY_MEMORY_SIZE': 10000,
        'LEARNING_RATE': 5e-4,
		'INITIAL_PERFORMANCE_THRSH': 100,

        'SCHEDULER': {
            'GAMMA': 0.1,
            'PERFORMANCE_LEAP': 50,
            'EPISODES_SUCCESS_SEQUENCE': 5,
            'SUCCESS_CRITERIA': "all_above_thresh",
            'MIN_EPISODES_BETWEEN_STEPS': 10,
        },

        'NUM_EPISODES': 50000,
        'OPT_LEVEL': 'O1',

        'CKPT_SAVE_FREQ': 10,
        'CKPT_PATH': '',
        'CKPT_SAVE_DIR': '',
        'CKPT_SAVE_BASE_DIR': '',

        'LOG': {
            'OUTPUT_DIR': '',
            'OUTPUT_BASE_DIR': '',
            'OUTPUT_FNAME': 'progress.txt',
            'EXP_NAME': '',
            'SAVE_FREQ': 1,
            'APPEND': True
        },

        'VALIDATION': {
            'Q_VALIDATION_FREQUENCY': 10,
            'SCORE_VALIDATION_FREQUENCY': 100,
            'SCORE_VALIDATION_SIZE': 10,
        },

        'VISUALIZE': False,
    }

    return default_params


cfg = CN(get_default_params())
