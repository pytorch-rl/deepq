from yacs.config import CfgNode as CN


def get_default_params():
    default_params = dict()

    default_params['PATHS'] = {}

    default_params['TRAIN'] = {
        'BATCH_SIZE': 128,
        'GAMMA': 0.999,
        'TARGET_UPDATE': 10
    }

    default_params['LOG'] = {}

    return default_params


cfg = CN(get_default_params())
