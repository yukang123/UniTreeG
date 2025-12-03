"""
From https://github.com/andrew-cr/tauLDR/blob/main/lib/utils/bookkeeping.py
"""
import ml_collections
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.utils.tensorboard as tensorboard
import glob
import os


def create_experiment_folder(save_location, inner_folder_name, include_time=True):
    today_date = datetime.today().strftime(r'%Y-%m-%d')
    now_time = datetime.now().strftime(r'%H-%M-%S')
    if include_time:
        total_inner_folder_name = now_time + '_' + inner_folder_name
    else:
        total_inner_folder_name = inner_folder_name
    path = Path(save_location).joinpath(today_date).joinpath(total_inner_folder_name)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint_dir, config_dir = create_inner_experiment_folders(path)

    return path, checkpoint_dir, config_dir


def create_inner_experiment_folders(path):

    checkpoint_dir = path.joinpath('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_archive_dir = checkpoint_dir.joinpath('archive')
    checkpoint_archive_dir.mkdir(exist_ok=True)

    config_dir = path.joinpath('config')
    config_dir.mkdir(exist_ok=True)

    return checkpoint_dir, config_dir


def save_config_as_yaml(cfg, save_dir):
    existing_configs = sorted(glob.glob(Path(save_dir).joinpath('config_*.yaml').as_posix()))

    if len(existing_configs) == 0:
        save_name = Path(save_dir).joinpath('config_001.yaml')
    else:
        most_recent_config = existing_configs[-1]
        most_recent_num = int(most_recent_config[-8:-5])
        save_name = Path(save_dir).joinpath('config_{0:03d}.yaml'.format(most_recent_num+1))

    with open(save_name, 'w') as f:
        yaml.dump(cfg.to_dict(), f)


def setup_tensorboard(save_dir, rank):
    if rank == 0:
        logs_dir = Path(save_dir).joinpath('tensorboard')
        logs_dir.mkdir(exist_ok=True)

        writer = tensorboard.writer.SummaryWriter(logs_dir)

        return writer
    else:
        return DummyWriter('none')


def save_checkpoint(checkpoint_dir, state, num_checkpoints_to_keep, ckpt_name=None):
    state_to_save = {
        'model': state['model'].state_dict(),
        'optimizer': state['optimizer'].state_dict(),
        'n_iter': state['n_iter'],
        'epoch': state['epoch']
    }
    if not ckpt_name:
        ckpt_name = 'ckpt_{0:010d}.pt'.format(state['n_iter'])
    torch.save(state_to_save,
        checkpoint_dir.joinpath(ckpt_name)
    )
    all_ckpts = sorted(glob.glob(checkpoint_dir.joinpath('ckpt_*.pt').as_posix()))
    if len(all_ckpts) > num_checkpoints_to_keep:
        for i in range(0, len(all_ckpts) - num_checkpoints_to_keep):
            os.remove(all_ckpts[i])


def save_archive_checkpoint(checkpoint_dir, state, ckpt_name=None):
    save_checkpoint(checkpoint_dir.joinpath('archive'), state, 9999999, ckpt_name)


def load_ml_collections(path):
    with open(path, 'r') as f:
        raw_dict = yaml.safe_load(f)
    return ml_collections.ConfigDict(raw_dict)


class DummyWriter():
    def __init__(self, save_dir):
        pass

    def add_scalar(self, name, value, idx):
        pass

    def add_figure(self, name, fig, idx):
        pass

    def close():
        pass


def set_in_nested_dict(nested_dict, keys, new_val):
    """
        Sets a value in a nested dictionary (or ml_collections config)
        e.g.
        nested_dict = \
        {
            'outer1': {
                'inner1': 4,
                'inner2': 5
            },
            'outer2': {
                'inner3': 314,
                'inner4': 654
            }
        } 
        keys = ['outer2', 'inner3']
        new_val = 315
    """
    if len(keys) == 1:
        nested_dict[keys[-1]] = new_val
        return
    return set_in_nested_dict(nested_dict[keys[0]], keys[1:], new_val)


def remove_module_from_keys(dict):
    # dict has keys of the form a.b.module.c.d
    # changes to a.b.c.d
    new_dict = {}
    for key in dict.keys():
        if '.module.' in key:
            new_key = key.replace('.module.', '.')
            new_dict[new_key] = dict[key]
        else:
            new_dict[key] = dict[key]

    return new_dict