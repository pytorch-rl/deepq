import yaml
from yacs import CfgNode as CN

def dict_from_yaml(cfg_path):
    with open(cfg_path) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg_dict


def load_from_yaml(path, yacs_cfg):
    cfg_dict = dict_from_yaml(path)
    tmp = CN(cfg_dict)
    yacs_cfg.merge_from_other_cfg(tmp)
