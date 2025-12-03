####################################################################################
### Adapted from:
### https://github.com/andrew-cr/tauLDR/blob/main/lib/utils/bookkeeping.py
####################################################################################
# Import public modules
import os
import torch.utils.tensorboard as tensorboard
from datetime import datetime
from pathlib import Path

def create_run_folder(save_location:str, 
                      inner_folder_name:str, 
                      include_time:bool=True,
                      folder_remark:str="") -> os.PathLike:
    """
    Create a run folder of the form 
    <save_location>/<datetime>/<inner_folder_name>
    and return the path to it.

    Args:
        save_location (str): Base save location of the folder.
        inner_folder_name (str): Folder name of the run.
        include_time (bool): Should the time be included in
            the <datetime> folder or not?
            (Default: False)

    Return:
        (os.PathLike): Path to the generated run folder.
    
    """
    today_date = datetime.today().strftime(r'%Y-%m-%d')
    now_time = datetime.now().strftime(r'%H-%M-%S')
    if include_time:
        total_inner_folder_name = now_time + '_' + inner_folder_name
    else:
        total_inner_folder_name = inner_folder_name
    path = Path(save_location).joinpath(today_date + folder_remark).joinpath(total_inner_folder_name)
    path.mkdir(parents=True, exist_ok=True)

    # Create the sub-folders within the run folder
    create_run_sub_folders(path)

    return path

def create_run_sub_folders(path:os.PathLike) -> None:
    """
    Create the sub-folders within the run folder.

    Args:
        path (os.PathLike): Path to run-folder in which 
            the sub-folders should be created in.
    """
    checkpoints_dir = path.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    checkpoints_archive_dir = checkpoints_dir.joinpath('archive')
    checkpoints_archive_dir.mkdir(exist_ok=True)

    configs_dir = path.joinpath('configs')
    configs_dir.mkdir(exist_ok=True)

    figs_save_dir = path.joinpath('figs_saved')
    figs_save_dir.mkdir(exist_ok=True)

    models_save_dir = path.joinpath('models_saved')
    models_save_dir.mkdir(exist_ok=True)

def setup_tensorboard(save_dir:str, 
                      rank:int) -> object:
    """
    Setup a tensorboard writer.

    Args:
        save_dir (str): Base save directory in which the
            the log-files for tensorboard should be saved 
            (i.e., written) to.
        rank (int): Process ID.

    Return:
        (object): Writter object.
        
    """
    if rank == 0:
        logs_dir = Path(save_dir).joinpath('tensorboard')
        logs_dir.mkdir(exist_ok=True)

        writer = tensorboard.writer.SummaryWriter(logs_dir)

        return writer
    else:
        return DummyWriter('none')
    
class DummyWriter():
    """
    Define a dummy (tensorboard) writter that can be called
    as tensorboard.writer.SummaryWriter but does nothing.
    """
    def __init__(self, save_dir):
        pass

    def add_scalar(self, name, value, idx:int):
        pass

    def add_figure(self, name, fig, idx):
        pass

    def close():
        pass