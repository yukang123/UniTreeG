# Import public modules
import ml_collections
import os
import torch
import numpy as np
from numbers import Number
from numpy.typing import ArrayLike
from pathlib import Path
from typing import Optional
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

# Import custom modules
from . import utils
from src import fm_utils, digress_utils

class DFMManager(object):
    """
    Manager used to manage models (either for training or inference/generation).
    """
    def __init__(self, 
                 cfg:ml_collections.ConfigDict, 
                 denoising_model:torch.nn.Module, 
                 denoising_optimizer:Optional[torch.optim.Optimizer]=None, 
                 predictor_models_dict:dict={}, 
                 logger:Optional[object]=None) -> None:
        """
        Args:
            cfg (ml_collections.ConfigDict): Config dictionary.
            denoising_model (torch.nn.Module): Denoising model object.
            denoising_optimizer (None or torch.optim.Optimizer): Torch optimizer
                object for the denoising model or None. 
                If None, no optimizer is used.
                (Default: None)
            predictor_models_dict (dict): Dictionary of the form
                {
                    '<predictor-model-name>': {
                        'model': <predictor-model-object>, 
                        'optimizer': <predictor-model-optimizer>
                    }, 
                    ...
                }
                with one entry per predictor model.
                Remark: <predictor-model-optimizer> can also be None (i.e., no optimizer
                        is defined for the corresponding predictor model).
            logger (None or object): Optional logger object or None. 
                If None, no logger is used.
                (Default: None)
        
        """
        self.cfg        = cfg                # Config dictionary
        self.D          = cfg.data.shape     # Number of dimensions
        self.S          = cfg.data.S         # State space
        self.mask_index = self.S-1           # Use the last state (with zero-based state index self.S-1) for the mask state
        self.pad_index  = cfg.data.pad_index # Extract eh state-index of the pad state
        self.fix_pads   = cfg.fix_pads       # Should pads be fixed during training?
        self.logger     = logger             # Logger object (or None)
        
        # Discrete or continuous time framework?
        self.num_timesteps = cfg.get('num_timesteps', None)
        if self.num_timesteps is None:
            # If no timesteps are defined, use the continuous time framework 
            # - Discrete Flow Modeling for denoising model
            # - Discrete Guidance for guidance
            self.continuous_time = True 
            self.display(f"Using the continuous-time framework (DFM for denoising model and DG for guidance).")
        else:
            # If timesteps are defined, use the discrete time framework 
            # - D3PM for denoising model
            # - DiGress for guidance
            self.continuous_time = False
            self.display(f"Using the discrete-time framework (D3PM for denoising model and DiGress for guidance) with {self.num_timesteps} timesteps.")
        
        if cfg.device is None:
            # Use 'cpu' as default device
            self.device = torch.device('cpu')
        else:
            self.device = cfg.device

        self.models_dict = {
            'denoising_model': denoising_model.to(self.device)
        }
        self.optims_dict = {
            'denoising_model': denoising_optimizer   
        }

        for predictor_model_name, predictor_model_specs in predictor_models_dict.items():
            self.models_dict[predictor_model_name] = predictor_model_specs["model"] if predictor_model_name == "oracle_predictor_model" else predictor_model_specs["model"].to(self.device)
            if 'optimizer' in predictor_model_specs:
                self.optims_dict[predictor_model_name] = predictor_model_specs['optimizer']
            else:
                self.optims_dict[predictor_model_name] = None
        
        #### ADDED ###
        self.predict_on_x1 = cfg.get('predict_on_x1', False)
        self.gumbel_norm_expectation, _, _ = fm_utils.get_gumbel_info(self.S)
        # aa = 1

    @property
    def predictor_models_dict(self) -> dict:
        """
        Return the predictor models dictionary.

        Return:
            (dict): Dictionary of the form
                {
                    '<predictor-model-name>': {
                        'model': <predictor-model-object>, 
                        'optimizer': <predictor-model-optimizer>
                    }, 
                    ...
                }
        """
        predictor_models_dict = dict()
        for model_name in self.models_dict:
            if model_name!='denoising_model':
                predictor_models_dict[model_name] = {
                    'model': self.models_dict[model_name],
                    'optimizer': self.optims_dict[model_name],
                }
        return predictor_models_dict

    @property
    def denoising_model(self) -> torch.nn.Module:
        """
        Return the denoising model.

        Return:
            (torch.nn.Module): Denoising model object.
        """
        return self.models_dict['denoising_model']

    def display(self, msg:str) -> None:
        """
        Display message either as logging info or as print if no logger has been defined. 
        
        Args:
            msg (str): Message to be displayed.
        
        """
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    def train(self) -> None:
        """ Set all models into train mode. """
        for model in self.models_dict.values():
            try:
                model.train()
            except:
                print(f"Model {model} does not have train() method.")

    def eval(self) -> None:
        """ Set all models into evaluation mode. """
        for model in self.models_dict.values():
            try:
                model.eval()
            except:
                print(f"Model {model} does not have eval() method.")

    def save_all_models(self, 
                        overwrite:bool=False) -> None:
        """
        Save all models managed by the manager.
        
        Args:
            overwrite (bool): Optional boolean flag that specifies if
                already saved model files (i.e., saved model weights) 
                should overwriten or not.
                (Default: False)
        
        """
        for model_name in self.models_dict:
            if model_name != "oracle_predictor_model":
                self.save_model(model_name, overwrite=overwrite)

    def load_all_models(self) -> None:
        """ Load all models. """
        for model_name in self.models_dict:
            if model_name != "oracle_predictor_model":
                self.load_model(model_name)
            # else:
            #     aa = 1

    def save_model(self, 
                   which_model:str, 
                   overwrite:bool=False) -> None:
        """
        Save the specified model (i.e., its weights).
        
        Args:
            which_model (str): Name of the to be saved model.
                This input must be one of the keys of self.models_dict.
            overwrite (bool): Optional boolean flag that specifies if
                already saved model files (i.e., saved model weights) 
                should overwriten or not.
                (Default: False)
        
        """
        # Check that the model exists
        if which_model not in self.models_dict:
            err_msg = f"'{which_model}' is not a valid model name. Valid model names are: {list(self.models_dict.keys())}"
            raise ValueError(err_msg)

        # Get the models save path and create a directory with its name
        # if it doesn't already exist
        models_save_dir = self.cfg.models_save_dir
        if os.path.isdir(models_save_dir)==False:
            os.mkdir(models_save_dir)
            self.display(f"Created directory: {models_save_dir}")

        # Construct the file path
        file_path = str(Path(models_save_dir, f"{which_model}.pt"))

        # If overwrite is False, check that the file does not already exist and throw an error otherwise
        if overwrite==False and os.path.isfile(file_path):
            err_msg = f"Model save file already exists, but overwriting is not permitted. File: {file_path}"
            raise ValueError(err_msg)

        # Save the model
        torch.save(self.models_dict[which_model].state_dict(), file_path)
        self.display(f"Saved '{which_model}' in: {file_path}")

    def load_model(self, 
                   which_model, 
                   file_path:Optional[str]=None) -> None:
        """
        Load a specific model (i.e., load its weights and assign them to the managed model).

        Args:
            which_model (str): Name of the to be loaded model.
                This input must be one of the keys of self.models_dict.
            file_path (None or str): Optional path to the folder in which 
                all model weights are saved in. 
                If None, use the path specified in the configs 
                (i.e., self.cfg.models_save_dir).
                (Default: None)
        
        """
        # Check that the model exists
        if which_model not in self.models_dict:
            err_msg = f"'{which_model}' is not a valid model name. Valid model names are: {list(self.models_dict.keys())}"
            raise ValueError(err_msg)

        # If the file_path is not passed, construct a 'default' one
        if file_path is None:
            # Get the model load directory
            if 'models_load_dir' in self.cfg:
                models_load_dir = self.cfg.models_load_dir
            else:
                self.display(f"'models_load_dir' not specified in configs, using the following save dir instead: {self.cfg.models_save_dir}")
                models_load_dir = self.cfg.models_save_dir

            # Construct the file path
            file_path = str(Path(models_load_dir, f"{which_model}.pt"))

        # If the file with the saved parameters does not exist, inform user and exit this method
        if os.path.isfile(file_path)==False:
            print(f"[Warning] Could not load the '{which_model}' because no save-file could be found for this model in: {file_path}")
            raise FileNotFoundError
            # if which_model == "denoising_model":
            #     raise FileNotFoundError #(f"Could not load the '{which_model}' because no save-file could be found for this model in: {file_path}")   
            # return

        # If the file path exists, load the model
        self.models_dict[which_model].load_state_dict(torch.load(file_path))
        self.display(f"Loaded '{which_model}' from: {file_path}")

    def train_on_batch(self, 
                       batch_data: object, 
                       which_model: str, 
                       training_state:dict={}) -> dict:
        """
        Train the specified model on a batch and return the updated
        'training_state' dictionary (that is passed as input).

        Args:
            batch_data (object): Batch data object (could be a 
                torch.tensor or some class object). 
            which_model (str): Name of the model to train on the batch.
                This input must be one of the keys of self.models_dict.
            training_state (dict): Dictionary holding information
                about the training state. This dictionary will be
                updated inside this method and returned.
                (Default: {})

        Return:
            (dict): Updated input dictionary 'training_state'.
        
        """
        if which_model not in self.models_dict:
            err_msg = f"The model '{which_model}' has not been defined."
            raise ValueError(err_msg)
        
        # Check that the optimizer is defined for the model
        if which_model not in self.optims_dict:
            err_msg = f"No optimizer defined for the model '{which_model}'."
            raise ValueError(err_msg)
        
        # Extract training configs
        clip_grad = self.cfg.training[which_model].clip_grad
        warmup    = self.cfg.training[which_model].warmup
        lr        = self.cfg.training[which_model].lr

        # Set the model into train mode
        self.models_dict[which_model].train()

        # Zero out gradients
        self.optims_dict[which_model].zero_grad()

        # Get the loss
        if which_model=='denoising_model':
            loss = self.get_denoising_batch_loss(batch_data=batch_data)
        else:
            loss = self.get_predictor_batch_loss(batch_data=batch_data, which_model=which_model)

        # Propagate gradients backward
        loss.backward()

        # Clip gradients if requrested
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(self.models_dict[which_model].parameters(), 1.0)
        
        # Apply warmup
        if warmup > 0:
            for g in self.optims_dict[which_model].param_groups:
                g['lr'] = lr * np.minimum(training_state['n_iter'] / warmup, 1.0)

        # Update model parameters
        self.optims_dict[which_model].step()

        # Update the loss in the training state
        training_state['n_iter'] += 1
        training_state['loss']    = loss.item()

        return training_state
    
    def eval_loss_on_batch(self, 
                           batch_data:object, 
                           which_model:str) -> dict:
        """
        Evaluate the loss of a specified model on a batch and return the
        loss over the batch as dictionary the form {'loss': <loss-value>}.

        Args:
            batch_data (object): Batch data object (could be a 
                torch.tensor or some class object). 
            which_model (str): Name of the model to evaluate the loss of 
                on the batch.
                This input must be one of the keys of self.models_dict.

        Return:
            (dict): Dictionary the form {'loss': <loss-value>} where <loss-value>
                is the loss of the specified model on the batch.
        
        """
        if which_model not in self.models_dict:
            err_msg = f"The model '{which_model}' has not been defined."
            raise ValueError(err_msg)
        
        # Set the model into evaluation mode
        self.models_dict[which_model].eval()

        # Get the loss
        if which_model=='denoising_model':
            loss = self.get_denoising_batch_loss(batch_data=batch_data)
        else:
            loss = self.get_predictor_batch_loss(batch_data=batch_data, which_model=which_model)

        return {'loss': loss.item()}
    
    def get_denoising_batch_loss(self, 
                                 batch_data:object) -> torch.tensor:
        """
        Return the loss of the denoising model on the batch.

        Args:
            batch_data (object): Batch data object (could be a 
                torch.tensor or some class object).

        Return:
            (torch.tensor): Loss of the denoising model on the
                batch as torch tensor. 
        
        """
        # Extract x1
        x1 = batch_data['x'] # (B, D)
        
        # Determine the loss
        if self.continuous_time:
            # Within the continuous-time framework, determine the (masking) DFM loss
            return fm_utils.flow_matching_loss_masking(denoising_model=self.models_dict['denoising_model'], 
                                                       x1=x1.long(),  # 'fm_utils.flow_matching_loss_masking' expects x1 containing long-type entries
                                                       mask_idx=self.mask_index, 
                                                       reduction='mean', 
                                                       pad_idx=self.pad_index, 
                                                       loss_mask=None) # No explicit loss-mask required
        else:
            # Within the discrete-time framework, determine the (masking) D3PM model loss
            return digress_utils.d3pm_loss_masking(denoising_model=self.models_dict['denoising_model'], 
                                                   x1=x1.long(),  # 'digress_utils.d3pm_loss_masking' expects x1 containing long-type entries
                                                   mask_idx=self.mask_index,
                                                   reduction='mean', 
                                                   pad_idx=self.pad_index,
                                                   loss_mask=None,
                                                   timesteps=self.num_timesteps)

    def get_predictor_batch_loss(self, 
                                 batch_data:object, 
                                 which_model:str) -> torch.tensor:
        """
        Return the loss of the specified property-predictor model on the batch.

        Args:
            batch_data (object): Batch data object (could be a 
                torch.tensor or some class object).
            which_model (str): Name of the model to return the loss 
                on the batch of.
                This input must be one of the keys of self.models_dict.

        Return:
            (torch.tensor): Loss of the specified property-predictor model 
                on the batch as torch tensor. 
        
        """
        # Extract x1
        x1 = batch_data['x'] # (B, D)

        # Get handle to predictor model and its y
        predictor_model = self.models_dict[which_model]
        y_name          = predictor_model.y_guide_name
        y_data          = batch_data[y_name]

        # Define the predictor-model log-probability as function f(x_{t}, t)=log[p(y=y*|x_{t}, t)]
        # with fixed/specified y=y* (=y_value)
        # Remark: xt is discrete here and not one-hot encoded (thus 'pass is_x_onehot=False')
        if self.predict_on_x1:
            predictor_log_prob_y = lambda y, x1: predictor_model.log_prob({y_name: y, 'x': x1}, is_x_onehot=False)
        else:
            predictor_log_prob_y = lambda y, xt, t: predictor_model.log_prob({y_name: y, 'x': xt}, t, is_x_onehot=False)

        if self.continuous_time:
            # Return loss for the continuous-time framework
            return fm_utils.predictor_loss_masking(predictor_log_prob_y, 
                                                   y_data, 
                                                   x1.long(), 
                                                   mask_idx=self.mask_index, 
                                                   reduction='mean', 
                                                   pad_idx=self.pad_index,
                                                   predict_on_x1=self.predict_on_x1)
        else:
            # Return loss for the discrete-time framework
            return digress_utils.d3pm_predictor_loss_masking(predictor_log_prob_y, 
                                                             y_data, 
                                                             x1.long(), 
                                                             mask_idx=self.mask_index, 
                                                             timesteps=self.num_timesteps, 
                                                             reduction='mean', 
                                                             pad_idx=self.pad_index)

    
    def get_t(self, 
              t_value:Number, 
              x1: torch.tensor) -> torch.tensor:
        """
        Broadcast a scalar t_value (castable to a float) to a 1D torch tensor
        that will correspond to a batched time tensor holding the same time 
        for each batch point.

        Args:
            t_value (castable to float): Scalar time value that should be
                used for each point of the batch.
            x1 (torch.tensor): Batch of (unnoised) spatial states as 2D 
                torch tensor of shape (B, D) where B is the batch size and 
                D is the spatial dimensionality of each batch point.
            
        Return:
            (torch.tensor): Batched time 1D torch tensor holding the same 
                t_value as time for each of the batch points. This tensor 
                will have shape (B,).
        
        """
        # Ensure that t_value is castable to a float in [0, 1]
        try:
            t_value = float(t_value)
        except ValueError:
            err_msg = f"Times must be castable to float in [0, 1], but got time {t_value} of type {type(t_value)}."
            raise TypeError(err_msg)
    
        # Check that t is in [0, 1]
        if (t_value<0) or (1<t_value):
            err_msg = f"Times must be be castable to float in [0, 1], but got time {t_value}."
            raise ValueError(err_msg)

        # Extract the batch size
        B = x1.shape[0]

        # Sample time t
        return t_value*torch.ones((B,)).to(self.device) # (B,)

    def sample_xt(self, 
                  x1:torch.tensor, 
                  t:torch.tensor) -> torch.tensor:
        """
        Sample xt. 
        
        Args:
            x1 (torch.tensor): Batch of (unnoised) spatial states as 2D 
                torch tensor of shape (B, D) where B is the batch size and 
                D is the spatial dimensionality of each batch point.
            t (torch.tensor): Batched time 1D torch tensor of shape (B,).

        Return:
            (torch.tensor): Sampled (i.e., noised) batched spatial states 
            as 2D torch tensor of shape (B, D).
        
        """
        # Extract the batch size
        B = x1.shape[0]

        # Sample x_t
        if self.continuous_time:
            # Sample x_t for continuous-time frameworks
            return fm_utils.sample_xt(x1, t, mask_idx=self.mask_index, pad_idx=self.pad_index)
        else:
            # Sample x_t for discrete-time frameworks
            return digress_utils.d3pm_sample_xt(x1, t, mask_idx=self.mask_index, timesteps=self.num_timesteps, pad_idx=self.pad_index)

    def generate(self, 
                 num_samples:int, 
                 seed:int=42, 
                 dt:float=0.001,
                 stochasticity:float=0.0,
                 predictor_y_dict:dict={},
                 grad_approx:bool=False,
                 guide_temp:float=1.0, # Guidance temperature
                 batch_size:int=500,
                 debug_predictor_form=False, # [DEBUG] only for debugging
                 debug_version="v1", # [DEBUG] only for debugging
                 debug_hyper=1.0, # [DEBUG] only for debugging
                 ) -> torch.tensor:
        """
        Generate x-samples.

        Remark: This method is a wrapper for the generation (i.e., sampling)
                methods defined in 'src.fm_utils' and 'src.digress_utils'.

        Args:
            num_samples (int): Number of samples to generate.
            seed (int): Seed for random state of generation.
                (Default: 42)
            dt (float): Time step for Euler method.
                (Default: 0.001)
            stochasticity (float): Stochasticity value used for continuous-time
                discrete-space flow models (i.e., '\eta').
                (Default: 0.0)
            predictor_y_dict (dict): Specification of property generation
                should be guided to in the form: 
                {'<property-model-name>': <specified-property-value>}.
                If this is an empty dictionary ({}), use unconditional generation.
                (Default: {} <=> i.e., unconditional generation)
            grad_approx (bool): Use Taylor-approximated gradients (TAG) for sampling
                (grad_approx=True), or exact sampling (grad_approx=False)?
                (Default: False)
            guide_temp (float): Guidance temperature.
                (Default: 1.0)
            batch_size (float): Number of samples to be generated in parallel.
                Remark: If batch_size<num_samples, multiple batches of samples will 
                    be generated and then concatenated to one torch tensor.
                    If num_samples<batch_size, the batch size will be set to num_samples.
                (Default: 500)

        Return:
            (torch.tensor) Generated x-samples as torch tensor of shape (num_samples, ...).

        """
        # Set all models into evaluation mode
        self.eval()

        # Set seed if passed
        if seed is not None:
            utils.set_random_seed(seed)

        # If predictor_y_dict is empty, do not use any guidance (thus set predictor_log_prob to None)
        guidance_at_x1 = self.cfg.sampler.get('guidance_at_x1', False) ## guided at x1
        if guidance_at_x1:
            assert self.predict_on_x1, "predict_on_x1 must be True when guidance_at_x1 is True"

        if len(predictor_y_dict)==0:
            predictor_log_prob = None
        elif len(predictor_y_dict)==1:
            predictor_model_name = list(predictor_y_dict.keys())[0]
            if predictor_model_name in self.models_dict:
                # Determine the predictor model, the name of the y-variable
                # this predictor model was trained on, and the y-value
                # specified to predictor-guide to.
                predictor_model = self.models_dict[predictor_model_name]
                y_name          = predictor_model.y_guide_name
                y_value         = predictor_y_dict[predictor_model_name]

                # Define the predictor-model log-probability as function f(x_{t}, t)=log[p(y=y*|x_{t}, t)]
                # with fixed/specified y=y* (=y_value)
                # Remarks: (1) y_value is a scalar (i.e. the same for all batch points), but log_prob expects 
                #              a tensorial y-input with a y-value per point in the batch (thus expand it).
                #          (2) When using Taylor-approximated guidance (TAG) is used (grad_approx=True), xt is one-hot
                #              encoded and used as input for the predictor log-probability (is_x_onehot=True).
                #              For exact guidance (grad_approx=False), xt is a passed as discrete state vector 
                #              to the predictor log-probability (is_x_onehot=False).
                #              => As grad_approx=True/is_x_onehot=True and grad_approx=False/is_x_onehot=False, 
                #                 one can pass grad_approx for is_x_onehot.
                
                if predictor_model_name == "oracle_predictor_model":
                    assert self.predict_on_x1, "oracle_predictor_model can only be used when predict_on_x1 is True"
                
                if debug_predictor_form and y_value == np.inf and self.predict_on_x1:
                    if debug_version == "v1":
                        predictor_log_prob = lambda x1: torch.log(torch.clamp(predictor_model({y_name: torch.tensor(y_value).to(x1.device).expand(x1.shape[0]), 'x': x1}, is_x_onehot=grad_approx if not guidance_at_x1 else False), min=1e-9))
                    elif debug_version == "v2":
                        predictor_log_prob = lambda x1: (predictor_model({y_name: torch.tensor(y_value).to(x1.device).expand(x1.shape[0]), 'x': x1}, is_x_onehot=grad_approx if not guidance_at_x1 else False)) ** 2
                    elif debug_version == "v3":
                        predictor_log_prob = lambda x1: (predictor_model({y_name: torch.tensor(y_value).to(x1.device).expand(x1.shape[0]), 'x': x1}, is_x_onehot=grad_approx if not guidance_at_x1 else False)) / debug_hyper
                    elif debug_version == "v4":
                        predictor_log_prob = lambda x1: predictor_model({y_name: torch.tensor(y_value).to(x1.device).expand(x1.shape[0]), 'x': x1}, is_x_onehot=grad_approx if not guidance_at_x1 else False) - debug_hyper
                else:
                    if y_value == np.inf: #or predictor_model_name == "oracle_predictor_model":
                        predictor_fn = predictor_model
                        # y_value = torch.nan
                    else:
                        predictor_fn = predictor_model.log_prob
                    ## [IMPORTANT] 
                    if self.predict_on_x1: 
                        predictor_log_prob = lambda x1: predictor_fn({y_name: torch.tensor(y_value).to(x1.device).expand(x1.shape[0]), 'x': x1}, is_x_onehot=grad_approx if not guidance_at_x1 else False)
                    else:
                        predictor_log_prob = lambda xt, t: predictor_fn({y_name: torch.tensor(y_value).to(xt.device).expand(xt.shape[0]), 'x': xt}, t, is_x_onehot=grad_approx)


            else:
                err_msg = f"The key '{predictor_model_name}' of input 'predictor_y_dict' does not correspond to the name of any of the (predictor) models: {list(self.models_dict.keys())}"
                raise ValueError(err_msg)
        else:
            err_msg = f"Guided generation is not implemented for more than 1 property."
            raise ValueError(err_msg)

        # Do flow-matching sampling
        if self.continuous_time:
            self.display("Sample using continuous-framework (DFM/DG).")
            # lower_threshold = self.cfg.sampler.get('low_threshold')
            # if lower_threshold is None:
            #     lower_threshold = 1e-9
            # else:
            #     while not isinstance(lower_threshold, float):
            #         try:
            #             lower_threshold = eval(lower_threshold.strip())
            #         except:
            #             print(f"Invalid value for 'low_threshold': {lower_threshold}", "threshold will be set to 1e-9")
            #             lower_threshold = 1e-9
            
            mcts = self.cfg.sampler.get('mcts', False)
            generation_kwargs = {"predict_on_x1": self.predict_on_x1, "guidance_at_x1": guidance_at_x1}
            generation_kwargs["verbose"] = self.cfg.sampler.get('verbose', False)
            if guidance_at_x1:
                if not mcts:
                    generation_kwargs.update({
                        "only_sample_for_rt_star": self.cfg.sampler.get('only_sample_for_rt_star', False),
                        "only_sample_for_unmasking": self.cfg.sampler.get('only_sample_for_unmasking', False),
                        "sample_k_for_guided_matrix": self.cfg.sampler.get('sample_k_for_guided_matrix', 10),
                        "log_Rt_ratio_temp": self.cfg.sampler.get('log_Rt_ratio_temp', 1.0),

                    })
                else:
                    generation_kwargs.update({
                        "mcts": mcts,
                        "branch_out_size": self.cfg.sampler.get('branch_out_size', 1),
                        "active_set_size": self.cfg.sampler.get('active_set_size', 1),
                        "svdd": self.cfg.sampler.get("svdd", False), 
                        "svdd_temp": self.cfg.sampler.get("svdd_temp", 1.0),
                    })
                generation_kwargs.update({ ### only for debugging
                    "without_guidance": self.cfg.sampler.get('without_guidance', False),
                    "add_remask_rate": self.cfg.sampler.get('add_remask_rate', True),
                })
            else:
                generation_kwargs.update({
                    "guidance_start_step": self.cfg.sampler.get('guidance_start_step', 0), # when to start using guidance
                    "guidance_end_step": self.cfg.sampler.get('guidance_end_step', -1), # when to stop using guidance
                })
                if self.predict_on_x1 and predictor_log_prob is not None:
                    generation_kwargs.update({
                        "predict_on_x1": self.predict_on_x1,
                        "sample_k_for_prob_xt_estimation": self.cfg.sampler.get('sample_k_for_prob_xt_estimation', 10),
                        "low_threshold": self.cfg.sampler.get('low_threshold', 1e-8),
                    })
                    if mcts: ## 
                        generation_kwargs.update({
                            "mcts": mcts,
                            ### [TO BE NOTICED] this is different from the default value in the original function
                            "use_guided_rate": self.cfg.sampler.get('use_guided_rate', False), 
                            "branch_out_size": self.cfg.sampler.get('branch_out_size', 1),
                            "active_set_size": self.cfg.sampler.get('active_set_size', 1),
                            "svdd": self.cfg.sampler.get("svdd", False), 
                            "svdd_temp": self.cfg.sampler.get("svdd_temp", 1.0),
                            "tds": self.cfg.sampler.get("tds", False),  
                            "tds_return_all": self.cfg.sampler.get("tds_return_all", False),  # whether to return all samples for TDS
                            "tds_reweight_temp": self.cfg.sampler.get("tds_reweight_temp", 1.0), # temperature for reweighting
                            "tds_prob_ratio_temp": self.cfg.sampler.get("tds_prob_ratio_temp", 1.0), # temperature for reweighting
                        })
                    else:
                        generation_kwargs.update({
                            "use_grad_fn_v1": self.cfg.sampler.get('use_grad_fn_v1', False),
                            "gumbel_softmax_t": self.cfg.sampler.get('gumbel_softmax_t', 1.0),
                            #### only for debugging
                            "sample_max": self.cfg.sampler.get("sample_max", False),
                        })
                # else:
                sample_xt_with_gumbel_max = self.cfg.sampler.get("sample_xt_with_gumbel_max", False)
                if sample_xt_with_gumbel_max:
                    generation_kwargs.update({
                        "sample_xt_with_gumbel_max": sample_xt_with_gumbel_max,
                        "strength_rescale": self.cfg.sampler.get("strength_rescale", False),
                        "gamma1": self.cfg.sampler.get("gamma1", 1.0),
                        "gamma2": self.cfg.sampler.get("gamma2", 1.0),
                        "add_gumbel_mean": self.cfg.sampler.get("add_gumbel_mean", False),
                        "rescale_method": self.cfg.sampler.get("rescale_method", "new"),
                        "strength_rescale_after_combination": self.cfg.sampler.get("strength_rescale_after_combination", False),
                        "gumbel_norm_expectation": self.gumbel_norm_expectation,
                    })
            print("generation kwargs", generation_kwargs)

            samples, time_info_dict = fm_utils.flow_matching_sampling(num_samples=num_samples, 
                                                   denoising_model=self.models_dict['denoising_model'],
                                                   S=self.cfg.data.S,
                                                   D=self.cfg.data.shape,
                                                   device=self.cfg.device,
                                                   dt=dt, 
                                                   mask_idx=self.mask_index,
                                                   pad_idx=self.pad_index,
                                                   batch_size=batch_size,
                                                   predictor_log_prob=predictor_log_prob, 
                                                   # Do not use any conditional denoising model 
                                                   # (i.e. no predict-free guidance):
                                                   cond_denoising_model=None,
                                                   guide_temp=guide_temp, 
                                                   stochasticity=stochasticity,
                                                   # Use Taylor-approximated guidance if 
                                                   # grad_approx=True, else don't:
                                                   use_tag=grad_approx,
                                                   argmax_final=self.cfg.sampler.argmax_final,
                                                   max_t=self.cfg.sampler.max_t,
                                                   x1_temp=self.cfg.sampler.x1_temp,
                                                   do_purity_sampling=self.cfg.sampler.do_purity_sampling,
                                                   purity_temp=self.cfg.sampler.purity_temp,
                                                   # 'train_num_tokens_freq_dict' is the distribution of the 
                                                   # number of unpadded tokens in the training set:
                                                   num_unpadded_freq_dict=self.cfg.data.get('train_num_tokens_freq_dict', None),
                                                   eps = 1e-9,                          
                                                   ############
                                                 # [IMPORTANT] Please add the below parameters in the new tasks
                                                    # guidance_at_x1 = guidance_at_x1,
                                                    **generation_kwargs
                                                )
            return samples, time_info_dict
        else:
            ### [WARNING] NEW METHODS MAY NOT APPLY THIS ###
            self.display("Sample using discrete-framework (D3PM/DiGress).")
            self.display("Remark: The input arguments 'dt' and 'stochasticity' are not used in this framework.")
            # In case of guidance, DiGress uses the 'gradient-approximation', thus check that the user specified this right
            # Remark: If predictor_y_dict={} (default) we are using unconditional generation and the grad_approx does not matter.
            if predictor_y_dict!={} and grad_approx==False:
                err_msg = f"Guided generation using the discrete-framework (i.e. with DiGress) requires the gradient approximation, but 'grad_approx=False' has been passed to the method 'generate'. Pass 'grad_approx=True' instead.."
                raise ValueError(err_msg)

            return digress_utils.d3pm_sampling(denoising_model=self.models_dict['denoising_model'], 
                                               num_samples=num_samples, # For this method, batch_size corresponds to num_samples
                                               S=self.cfg.data.S, 
                                               D=self.cfg.data.shape,
                                               device=self.cfg.device, 
                                               timesteps=self.num_timesteps,
                                               mask_idx=self.mask_index, 
                                               pad_idx=self.pad_index, 
                                               batch_size=batch_size,
                                               predictor_log_prob=predictor_log_prob, 
                                               guide_temp=guide_temp,
                                               x1_temp=self.cfg.sampler.x1_temp,
                                               # 'train_num_tokens_freq_dict' is the distribution of the 
                                               # number of unpadded tokens in the training set:
                                               num_unpadded_freq_dict=self.cfg.data.get('train_num_tokens_freq_dict', None))
        
    def predict_property(self, 
                         predictor_model_name:str, 
                         x:torch.tensor, 
                         t:torch.tensor, 
                         return_probs:bool=True) -> ArrayLike:
        """
        Predict the property (output of the specified predictor model) for
        batched input spatial states x and times t.

        Args:
            predictor_model_name (str): Name of the predictor model to be
                used for prediction and that also specifies which property 
                is therefore predicted.
                This input must be one of the keys of self.models_dict.
            x (torch.tensor): Batch of spatial states as 2D torch tensor of 
                shape (B, D) where B is the batch size and D is the spatial 
                dimensionality of each batch point.
            t (torch.tensor): Batched time 1D torch tensor of shape (B,).
            return_probs (bool): In case that the output type of the specified
                predictor model is 'class distribution', one can also return 
                the predicted class probabilities instead of the predicted 
                property values.
                If return_probs=True, these predicted class probabilities are 
                returned instead of the predicted property values.
            
        Return:
            (numpy.ndarray or torch.tensor): Predicted property values
                (or probabilities) as batched array like object.

        """
                        
        # Check that the predictor model has been defined
        if predictor_model_name not in self.models_dict:
            err_msg = f"No predictor model with name '{predictor_model_name}' specified in self.models_dict. The specified predictor models are: {list(self.models_dict.keys())}"
            raise ValueError(err_msg)
        
        # Set the predictor model into evaluatiion model
        if predictor_model_name != "oracle_predictor_model":
            self.models_dict[predictor_model_name].eval()

        # Setup x and t for the prediction model and predict
        _x         = torch.tensor(x).reshape(1, -1).to(self.device)
        _t         = torch.tensor([t]).to(self.device)
        prediction = utils.to_numpy(self.models_dict[predictor_model_name]({'x': _x}, _t)).squeeze()

        # Differ cases if the predictor output type is a 'class distribution'
        if self.models_dict[predictor_model_name].output_type=='class_distribution':
            # Differ the case where we want to return a probability vector (return_probs=True)
            # and where we return the most likely class (return_probs=False)
            if return_probs:
                return prediction
            else:
                return np.argmax(prediction, axis=-1)
        else:
            return prediction



