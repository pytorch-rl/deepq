from yacs.config import CfgNode as CN


def get_default_params():
    default_params = dict()

    default_params['PATHS'] = {}

    default_params['TRAIN'] = {
        'BATCH_SIZE': 128,
        'GAMMA': 0.999,
        'TARGET_UPDATE': 10,
        'EPS_START': 0.9,
        'EPS_END':  0.05,
        'EPS_DECAY': 200,
    }

    default_params['LOG'] = {}

    return default_params


cfg = CN(get_default_params())
