# Import public modules
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional

# Import custom modules
from . import utils

def train_model(manager:object, 
                train_dataloader:object, 
                which_model:str='denoising', 
                validation_dataloader:Optional[object]=None, 
                num_epochs:int=10, 
                random_seed:Optional[int]=None, 
                display_info_every_nth_epoch:int=1, 
                plot_training_curves:bool=False,
                figs_save_dir:Optional[str]=None,
                tensorboard_writer:Optional[object]=None,
                logger:Optional[object]=None) -> None:
    """
    Train one of the model's of the diffusion manager specified by 'which_model'.
    
    Args:
        manager (object):
            Manager that contains the denoising-model and abstracts its 
            functionality. Also contains predictor-model used for guidance.
        train_dataloader (object): (torch) dataloader object of the train set.
        which_model (str): Which model to train 'denoising' or the name of 
            the property models.
        valid_dataloader (None or object): If not None, (torch) dataloader
            object of the validation set.
            (Default: None)
        num_epochs (int): Number of epochs to train for.
            (Default: 10)
        random_seed(int): Random seed to be used for training.
            (Default: 42)
        display_info_every_nth_epoch (int): Display information every nth epoch.
            (Default: 1)
        plot_training_curves (bool): Should the training curves (e.g. loss vs. epoch)
            be plotted or not?
            (Default: False)
        figs_save_dir (None or str): Path to directory where the figures should be saved 
            in as string. If None, do not save figures.
            (Default: None)
        tensorboard_writer (None or object): Writer object to write log-files 
            for tensorboard. If None, do not write any log-files for tensorboard.
            (Default: None) 
        logger (None or object): Logger object. If None, do not use a logger.
            (Default: None)
    
    Return:
        None
    
    """
    # Check that the model is model of the manager
    if which_model not in manager.models_dict:
        err_msg = f"'{which_model}' is not the name of a model in the manager. The model names are: {list(manager.models_dict.keys())}"
        raise ValueError(err_msg)

    # Set a random seed if it was passed (i.e. not None)
    if random_seed is None:
        if logger is None:
            print("No random seed passed, thus no seed is set.")
        else:
            logger.info("No random seed passed, thus no seed is set.")
    else:
        utils.set_random_seed(random_seed)

    # Notify user
    if logger is None:
        print(f'Model: {which_model}')
        print(f'Number parameters: {manager.models_dict[which_model].num_parameters}')
        print(f"Training for {num_epochs} epochs on device '{manager.models_dict[which_model].device}':")
    else:
        logger.info(f'Model: {which_model}')
        logger.info(f'Number parameters: {manager.models_dict[which_model].num_parameters}')
        logger.info(f"Training for {num_epochs} epochs on device '{manager.models_dict[which_model].device}':")

    # Loop over the epochs
    epoch_train_loss_list = list()
    epoch_valid_loss_list = list()
    training_state = dict(n_iter=0)
    start_time = time.time()
    for epoch in range(num_epochs):
        batch_train_loss_list = list()
        for batch_train_data in train_dataloader:
            # Train on the batch for the specified model
            training_state = manager.train_on_batch(
                batch_train_data, 
                which_model, 
                training_state=training_state)
        
            batch_train_loss_list.append(training_state['loss'])

        # Average the train losses in the batch thereby determining the total train
        # loss of the current epoch and append this to the epoch train loss list.
        # Remarks: (1) This is a moving loss (i.e. different batches have seen
        #              different parameters due to parameter updates between them).
        #          (2) Averaging is allowed if the batch loss is intensive (i.e. 
        #              calculated per batch point).
        epoch_train_loss = np.mean(np.array(batch_train_loss_list))
        epoch_train_loss_list.append(epoch_train_loss)

        # Determine the loss on the validation set for the epoch if the 
        # validation dataloader has been defined (i.e. is not None)
        if validation_dataloader is None:
            epoch_valid_loss = None
        else:
            batch_valid_loss_list = list()
            for batch_valid_data in validation_dataloader:
                # Determine the validation loss on the batch for the specified model
                # WITHOUT training the model on this batch
                eval_output = manager.eval_loss_on_batch(
                    batch_valid_data, 
                    which_model)
                batch_valid_loss_list.append(eval_output['loss'])

            # Average the validation losses in the batch thereby determining the total validation
            # loss of the current epoch and append this to the epoch validation loss list.
            # Remarks: Averaging is allowed if the batch loss is intensive 
            #          (i.e. calculated per batch point).
            epoch_valid_loss = np.mean(np.array(batch_valid_loss_list))
            epoch_valid_loss_list.append(epoch_valid_loss)

        # Display information about the current epoch if requested for it
        if epoch%display_info_every_nth_epoch==0:
            if logger is None:
                print(f"[{epoch}] (train-moving-loss) | {epoch_train_loss}")
                print(f"[{epoch}] (validation-loss)   | {epoch_valid_loss}")
            else:
                logger.info(f"[{epoch}] (train-moving-loss) | {epoch_train_loss}")
                logger.info(f"[{epoch}] (validation-loss)   | {epoch_valid_loss}")

        # Write to tensorboard if a writter was passed
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar(f"{which_model}/Train-moving-loss", epoch_train_loss, epoch)
            tensorboard_writer.add_scalar(f"{which_model}/Validation-loss", epoch_valid_loss, epoch)

    if logger is None:
        print(f"Training done (Duration: {(time.time()-start_time)/60:.2f} mins)")
    else:
        logger.info(f"Training done (Duration: {(time.time()-start_time)/60:.2f} mins)")

    # Plot the losses of the epochs
    if plot_training_curves and 1<len(epoch_train_loss_list):
        fig = plt.figure(figsize=(6, 6))
        plt.plot(epoch_train_loss_list, 'b-', label='Train (moving) loss')
        if 0==len(epoch_valid_loss_list):
            plt.title(f"(Moving) train loss of final episode: {epoch_train_loss_list[-1]:.5f}")
            y_min = min([0, np.min(epoch_train_loss_list)*1.1])
            y_max = np.max(epoch_train_loss_list[1:])*1.1
        else:
            optimal_epoch = int(np.argmin(epoch_valid_loss_list))
            plt.title(f"Minimal validation loss: {min(epoch_valid_loss_list):.5f} (@epoch: {optimal_epoch})")
            plt.plot(epoch_valid_loss_list, 'r-', label='Validation loss')
            plt.legend()
            y_min = min([0, np.min(epoch_train_loss_list)*1.1, np.min(epoch_valid_loss_list)*1.1])
            y_max = max([np.max(epoch_train_loss_list[1:]), np.max(epoch_valid_loss_list[1:])])*1.1
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([y_min, y_max])
        plt.show()

        if figs_save_dir is not None:
            file_path = str(Path(figs_save_dir, f"training_curve_{which_model}"))
            fig.savefig(file_path)