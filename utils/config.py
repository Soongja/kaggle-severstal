import os
import shutil
import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    c.DATA_DIR = ''
    c.TRAIN_DF = ''
    c.FOLD_DF = ''
    c.PSEUDO_TRAIN_DF = ''
    c.PSEUDO_FOLD_DF = ''
    c.TRAIN_ALL = False
    c.SAMPLER = ''
    c.GEM = False

    c.TRAIN_DIR = ''

    c.PARALLEL = False
    c.DEBUG = False
    c.DEBUG_IMAGE = False
    c.PRINT_EVERY = 1

    c.TRAIN = edict()
    c.TRAIN.BATCH_SIZE = 4
    c.TRAIN.NUM_WORKERS = 8
    c.TRAIN.NUM_EPOCHS = 100

    c.EVAL = edict()
    c.EVAL.BATCH_SIZE = 8
    c.EVAL.NUM_WORKERS = 8

    c.DATA = edict()
    c.DATA.IMG_H = 256
    c.DATA.IMG_W = 1600

    c.MODEL = edict()
    c.MODEL.ARCHITECTURE = ''
    c.MODEL.ENCODER = ''
    c.MODEL.NAME = ''
    c.MODEL.ADD_FC = False

    c.LOSS = edict()
    c.LOSS.NAME = 'bce'
    c.LOSS.FINETUNE_EPOCH = 10000
    c.LOSS.FINETUNE_LOSS = ''
    c.LOSS.PARAMS = edict()

    c.OPTIMIZER = edict()
    c.OPTIMIZER.NAME = 'adam'
    c.OPTIMIZER.LR = 0.001
    c.OPTIMIZER.PARAMS = edict()

    c.SCHEDULER = edict()
    c.SCHEDULER.NAME = 'none'
    c.SCHEDULER.PARAMS = edict()

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config_path, train_dir):
    configs = [config for config in os.listdir(train_dir)
               if config.startswith('config') and config.endswith('.yml')]
    if not configs:
        last_num = -1
    else:
        last_config = list(sorted(configs))[-1]  # ex) config5.yml
        last_num = last_config.split('.')[0].split('config')[-1]

    save_name = 'config' + str(int(last_num) + 1) + '.yml'

    shutil.copy(config_path, os.path.join(train_dir, save_name))
