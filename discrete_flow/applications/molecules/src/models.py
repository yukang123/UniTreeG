# Import public modules
import ml_collections
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Import custom modules
from . import utils
# from src.fm_utils import PREDICT_ON_X1
# PREDICT_ON_X1 = False
from applications.molecules.scripts.utils import PREDICT_ON_X1
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
from rdkit import Chem
from rdkit.Chem import QED
from .cheminf import get_property_value, get_property_value_all
from .data_handling import SMILESEncoder
import tdc
import time
import os
# Define the denoising model
class DenoisingModel(torch.nn.Module):
    def __init__(self, 
                 cfg:ml_collections.ConfigDict, 
                 time_encoder:object, 
                 logger:Optional[object]=None) -> None:
        """
        Args:
            cfg (ml_collections.ConfigDict): Config dictionary.
            time_encoder (object): Time encoder object.
            logger (None or object): Optional logger object.
                If None, no logger is used.
                (Default: None)

        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.D           = cfg.data.shape
        self.S           = cfg.data.S
        self.pad_index   = cfg.data.pad_index
        self.num_hidden  = cfg.denoising_model.num_hidden # Number of hidden layers
        self.hidden_dim  = cfg.denoising_model.hidden_dim # Dimension of each hidden layer
        self.p_dropout   = cfg.denoising_model.p_dropout
        self.eps         = float(cfg.denoising_model.eps)
        self.stack_time  = cfg.denoising_model.stack_time
        self.logger      = logger
        
        # Construct self.hidden_dims as list with self.num_hidden number elements
        # where each element corresponds to self.hidden_dim
        self.hidden_dims = [self.hidden_dim]*self.num_hidden

        self.display(f"Stack time to x as denoising model input: {self.stack_time}")

        # Set class attributes
        self.encode_t = lambda t: time_encoder(t)
        
        # Define input dimension
        self.input_dim = self.D*self.S
        if self.stack_time:
            # If the time should be stacked, add the time 
            # encoding dimension to the input dimension
            self.input_dim += self.time_encoder.dim
        
        # Define the output dimension
        self.output_dim = self.D*self.S 

        # Define the linear parts of the model
        linear_list       = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        linear_list      += [torch.nn.Linear(self.hidden_dims[layer_id-1], self.hidden_dims[layer_id]) for layer_id in range(1, len(self.hidden_dims))]
        self.linear_list  = torch.nn.ModuleList(linear_list)
        self.linear_last  = torch.nn.Linear(self.hidden_dims[-1], self.output_dim)

        # Define an activation function
        self.activation_fn = activation_fn_factory(cfg.denoising_model)

        # Define (global) dropout function
        self.dropout_fn = torch.nn.Dropout(p=self.p_dropout, inplace=False)


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

    @property
    def device(self) -> object:
        """
        Return the device the model parameters are on.
        
        Returns:
            (object): The device the model parameters are on.
        
        """
        # Pick the first parameter and return on which device it is on
        return next(self.parameters()).device
    
    @property
    def num_parameters(self) -> int:
        """
        Return the number of model parameters.
        
        Source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/10

        Return:
            (int): Number of model parameters.

        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_x(self, x:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)

        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()

    def forward(self, 
                xt:torch.tensor, 
                t:torch.tensor,
                # is_x_onehot:bool=False,
                ) -> torch.tensor:
        """"
        Define forward pass of the model.
        
        Args:
            xt (torch.tensor): Shape (B, D).
            t (torch.tensor): Shape (B,).

        Return:
            (torch.tensor): Logits of shape (B, D, S).
        
        """
        # Extract the batch size
        B = xt.shape[0]

        # Encode space and flatten from (B, D, S) to (B, D*S)
        if xt.shape[-1] == self.S and xt.shape[-2] == self.D: #is_x_onehot:
            xt_enc = xt
        else:
            xt_enc = self.encode_x(xt) # (B, D, S)
        xt_enc = xt_enc.view(-1, self.D*self.S) # (B, D*S)

        # Stack encoded time to h if requested
        if self.stack_time:
            t_enc = self.encode_t(t)
            h = torch.cat([xt_enc, t_enc], dim=-1) 
        else:
            h = xt_enc

        # Perform pass through the network
        for layer_id in range(len(self.linear_list)):
            h = self.dropout_fn(h)
            h = self.linear_list[layer_id](h)
            h = self.activation_fn(h)

        # Shape (B, #classes)
        h = self.dropout_fn(h)
        h = self.linear_last(h)
    
        # Bring logits in correct shape
        logits = h.view(-1, self.D, self.S) # (B, D, S)
    
        return logits



import functools
def x0_predictor(predict_on_x1: bool=False):
    
    def class_decorator(cls):

        old_init = cls.__init__
        @functools.wraps(old_init)
        def new_init(self, **kwargs):
            if predict_on_x1:
                kwargs["model_cfg"].stack_time = False
                self.predict_on_x1 = True
            else:
                self.predict_on_x1 = False
            old_init(self, **kwargs)
        cls.__init__ = new_init

        if predict_on_x1:
            # @functools.wraps(cls.log_prob)
            old_log_prob = cls.log_prob
            def log_prob_wrapper(self, batch_data, is_x_onehot=False):
                t = torch.ones(batch_data['x'].shape[0], device=self.device)
                output = old_log_prob(self, batch_data, t, is_x_onehot=is_x_onehot)
                return output
            cls.log_prob = log_prob_wrapper

        return cls

    return class_decorator

@x0_predictor(predict_on_x1=PREDICT_ON_X1)
class DiscretePredictorGuideModel(torch.nn.Module):
    output_type = 'class_distribution'
    
    def __init__(self, 
                 num_classes:int, 
                 output_layer_dim:int, 
                 model_cfg:ml_collections.ConfigDict, 
                 general_cfg:ml_collections.ConfigDict, 
                 time_encoder:object, 
                 logger:Optional[object]=None) -> None:
        """
        Args:
            num_classes (int): Number of classes.
            output_layer_dim (int): Dimension of the output layer.
            model_cfg (ml_collections.ConfigDict): Model specific config dictionary.
            general_cfg (ml_collections.ConfigDict): General (non-model specific) config dictionary.
            time_encoder (object): Time encoder object.
            logger (None or object): Optional logger object.
                If None, no logger is used.
                (Default: None)

        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.D                = general_cfg.data.shape
        self.S                = general_cfg.data.S
        self.y_guide_name     = model_cfg.y_guide_name
        self.hidden_dims      = model_cfg.hidden_dims
        self.p_dropout        = model_cfg.p_dropout
        self.eps              = float(model_cfg.eps)
        self.stack_time       = model_cfg.stack_time
        self.y_enc_dim        = num_classes
        self.time_encoder     = time_encoder
        self.output_layer_dim = output_layer_dim
        self.logger           = logger

        self.display(f"Stack time to x as {self.y_guide_name}-predictor model input: {self.stack_time}")

        # Define input dimension
        self.input_dim = self.D*self.S
        if self.stack_time:
            # If the time should be stacked, add the time 
            # encoding dimension to the input dimension
            self.input_dim += self.time_encoder.dim

        # Define the t encoder
        self.encode_t = lambda t: self.time_encoder(t)

        # Define the model parts
        linear_list       = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        linear_list      += [torch.nn.Linear(self.hidden_dims[layer_id-1], self.hidden_dims[layer_id]) for layer_id in range(1, len(self.hidden_dims))]
        self.linear_list  = torch.nn.ModuleList(linear_list)
        self.linear_last  = torch.nn.Linear(self.hidden_dims[-1], self.output_layer_dim) # Output layer

        # Define an activation function
        self.activation_fn = activation_fn_factory(model_cfg)

        # Define (global) dropout function
        self.dropout_fn = torch.nn.Dropout(p=self.p_dropout, inplace=False)
    
    @property
    def device(self) -> object:
        """
        Return the device the model parameters are on.
        
        Returns:
            (object): The device the model parameters are on.
        
        """
        # Pick the first parameter and return on which device it is on
        return next(self.parameters()).device
    
    @property
    def num_parameters(self) -> int:
        """
        Return the number of model parameters.
        
        Source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/10

        Return:
            (int): Number of model parameters
            
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


    @property
    def num_categories(self) -> int:
        """
        Return the number of categories that correspond to the dimensionality
        of the one-hot encoded y.

        Return:
            (int): Number of categories.

        """
        return self.y_enc_dim

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

    def encode_x(self, x:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)

        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()
    
    def encode_y(self, y:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor y.

        Args:
            y (torch.tensor): Torch tensor of shape (B,) holding
                categorical property entries, where B is the batch 
                size.

        Return:
            (torch.tensor): Tensor where the categorical entries 
                (of which there are self.num_categoricals different one) 
                of y have been one-hot encoded to a tensor of shape 
                (B, self.num_categoricals).

        """
        return torch.nn.functional.one_hot(y.long(), num_classes=self.num_categories).float()

    def forward(self, 
                batch_data_t:dict, 
                t:torch.tensor, 
                is_x_onehot:bool=False) -> torch.tensor:
        """
        Define forward pass of the model.
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): Probabilities for each component of the encoded property 'y'
                as 2D torch tensor of shape (batch_size, dim[encoded(y)])
        
        """
        # Get xt and t, encode both, and stack them
        xt = batch_data_t['x']
        B  = xt.shape[0]

        # Differ the cases where x is already encoded or not
        if is_x_onehot:
            # xt is already encoded
            xt_enc = xt # (B, D, S)
        else:
            # xt has to be encoded
            xt_enc = self.encode_x(xt) # (B, D, S)

        # Flatten features
        xt_enc = xt_enc.view(-1, self.D*self.S) # (B, D*S)

        # Stack encoded time to h if requested
        if self.stack_time:
            t_enc = self.encode_t(t).view(-1, 1)
            h = torch.cat([xt_enc, t_enc], dim=-1) 
        else:
            h = xt_enc

        # Perform pass through the network
        for layer_id in range(len(self.linear_list)):
            h = self.dropout_fn(h)
            h = self.linear_list[layer_id](h)
            h = self.activation_fn(h)
        
        # Perform pass through last layer
        h = self.dropout_fn(h)
        h = self.linear_last(h) # (B, output_layer_dim)

        # Determine the class probabilities from the output layer
        p_y = self._get_p_y_from_output_layer(h, t)
        return p_y
    
    def _get_p_y_from_output_layer(self, 
                                   h:torch.tensor, 
                                   t:torch.tensor) -> torch.tensor:
        """
        Map the values from the output layer (i.e. self.last_linear) to
        class propabilities.

        Args:
            h (torch.tensor): Output layer values.
            t (torch.tensor): Times.
        
        Return:
            (torch.tensor): Class probabilities p(y|h(x),t).
        
        """
        raise NotImplementedError("The method '_get_p_y_from_output_layer' has not been implemented.")

    def log_prob(self, 
                 batch_data_t:torch.tensor, 
                 t:torch.tensor, 
                 is_x_onehot:bool=False) -> torch.tensor:
        """
        Return the log probability given the data. 
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): (Batched) log-probability for each point in the batch as
                1D torch tensor of shape (batch_size,).
        
        """
        y_data = batch_data_t[self.y_guide_name]
        xt = batch_data_t['x']

        # Encode the property input y (class index)
        # Shape (B, #classes)
        y_data_enc = self.encode_y(y_data).to(xt.device)

        # Determine the class-probabilities
        p_y_pred = self.forward(batch_data_t, t, is_x_onehot=is_x_onehot) # Shape (B, #classes)

        # Calculate the categorical log-probability for each point and each
        # category/feature and then sum over the feature (i.e. the second) axis
        log_prob = torch.sum(y_data_enc*torch.log(p_y_pred+self.eps), dim=-1)

        return log_prob
    
# @x0_predictor(predict_on_x1=PREDICT_ON_X1)
class CategoricalPredictorGuideModel(DiscretePredictorGuideModel):
    def __init__(self, 
                 num_categories:int, 
                 **kwargs) -> None:
        """
        Args:
            num_categories (int): Number of mutually exclusive categories.
            **kwargs: Keyword arguments forwarded to parent class.

        """

        # Initialize the parent class
        # Remark: For categorical, the dimension of the output layer 
        #         corresponds to the number of categories.
        super().__init__(num_classes=num_categories, output_layer_dim=num_categories, **kwargs)

        # Define the softmax function that should be applied along the
        # second axis (i.e. the feature axis)
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    def _get_p_y_from_output_layer(self, 
                                   h:torch.tensor, 
                                   t:torch.tensor) -> torch.tensor:
        """
        Map the values from the output layer (i.e. self.last_linear) to
        class propabilities.

        Remark:
        Here, use a softmax to transform output layer values h 
        (in R) to class probabilities of shape (B, #categories).

        Args:
            h (torch.tensor): Output layer values.
            t (torch.tensor): Times.
        
        Return:
            (torch.tensor): Class probabilities p(y|h(x),t).
        
        """
        return self.softmax_fn(h)
    
# @x0_predictor(predict_on_x1=PREDICT_ON_X1)
class OrdinalPredictorGuideModel(DiscretePredictorGuideModel):    
    def __init__(self, 
                 num_ordinals:int, 
                 sigma_noised:float=1.0,
                 **kwargs) -> None:
        """
        Args:
            num_ordinals (int): Number of ordinals.
            sigma_noised (float): Sigma of the properties 
                (for which a predictor is setup here) of 
                the fully noised samples (at t=0).
                (Default: 1.0)
            **kwargs: Keyword arguments forwarded to parent class.

        """
        # Initialize the parent class
        # Remark: For ordinal, the dimension of the output layer corresponds
        #         to 1 (mean of discretized normal distribution).
        super().__init__(num_classes=num_ordinals, output_layer_dim=1, **kwargs)

        # Define ordinal model specific attributes
        self.sigma_noised       = sigma_noised
        self.log_sigma_unnoised = torch.nn.Parameter(torch.log(torch.tensor(sigma_noised))) # Initialize equal to log(sigma_noised)

    @property
    def sigma_unnoised(self) -> torch.tensor:
        """
        Return the unnoised sigma based on the model parameter 'self.log_sigma_unnoised'.

        Return:
            (torch.tensor): Unnoised sigma.

        """
        return torch.exp(self.log_sigma_unnoised)

    def get_sigma_t(self, t:torch.tensor) -> torch.tensor:
        """
        Return sigma(t) as a linear interpolation between the
        noised sigma at t=0 and the unnoised sigma at t=1.
        
        Remark: 
        At t=1 we have the data distribution (unnoised)
        and at t=0 we have the noised distribution.
        Thus, interpolate the predictor sigma in a similar way.

        Args:
            t (torch.tensor): Times.

        Return:
            (torch.tensor): Determined sigma(t).
        
        """
        return t*self.sigma_unnoised+(1-t)*self.sigma_noised
    
    def plot_sigma_t(self) -> object:
        """
        Plot sigma(t) vs. t and return the resulting figure.
        
        Return:
            (object): Matplotlib figure object.
        
        """
        fig = plt.figure()
        t = torch.linspace(0, 1, 1000).to(self.sigma_unnoised.device)
        sigma_t = self.get_sigma_t(t)
        plt.plot(utils.to_numpy(t), utils.to_numpy(sigma_t), 'b-')
        plt.xlabel('t')
        plt.ylabel('sigma(t)')
        plt.xlim([0, 1])
        plt.ylim([0, max(utils.to_numpy(sigma_t))*1.05])
        plt.show()
        return fig

    def _get_p_y_from_output_layer(self, 
                                   mu:torch.tensor,
                                   t:torch.tensor) -> torch.tensor:
        """
        Map the values from the output layer (i.e. self.last_linear) to
        class propabilities.

        Remark:
        Here, the output layer values correspond to the mean 'mu' of a discretized 
        normal distribution that is used as ordinal distribution.
        Thus, calculate the class probabilities by integrating the normal distribution
        within the ordinal bounaries.
        The standard deviation (sigma) is determined from the passed time 't' (see below).

        Args:
            mu (torch.tensor): Output layer values (that determines to the mean of a 
                discretized normal distribution) of shape (B, 1).
            t (torch.tensor): Times.
        
        Return:
            (torch.tensor): Class probabilities p(y|mu(x),t).
        
        """
        # Define the boundaries of the ordinals
        # Remark: The ordinal indices are zero based ([0, 1, ..., num_ordinals-1]) so
        #         that the ordinal boundaries are [-0.5, 0.5, ..., (num_ordinals-1)+-0.5]
        #         which is equivalent to [-0.5, 0.5, ..., num_ordinals-0.5]
        num_ordinals = self.y_enc_dim
        ordinal_bounds = torch.linspace(-0.5, num_ordinals-0.5, num_ordinals+1).to(mu.device) # (#ordinals+1, )

        # Determine sigma(t)
        sigma_t = (self.get_sigma_t(t)+self.eps).reshape(-1, 1) # (B, 1)
       
        # Determine the intergrals of the normal distribution defined by h up to 
        # each of the ordinal boundaries
        cdfs = torch.distributions.normal.Normal(loc=mu, scale=sigma_t).cdf(ordinal_bounds) # (B, num_ordinals+1)
       
        # Determine the probability of each ordinal as the integral between each
        # of its boundaries
        ordinal_ints = cdfs[:, 1:]-cdfs[:, :-1] # (B, num_ordinals)
        
        # These integrals do not sum up to 1, because the contribution of the integrals over the normal 
        # distribution for -inf to -0.5 (first lower-bound) and from num_ordinals-0.5 (last upper-bound) is 
        # not included (and because only the integral from -inf to inf gives 1).
        # Thus, normalize these integrals to obtain the ordinal probabilities
        ordinal_probs = ordinal_ints/(torch.sum(ordinal_ints, dim=-1).reshape(-1, 1)+self.eps) # (B, num_ordinals)

        return ordinal_probs

@x0_predictor(predict_on_x1=PREDICT_ON_X1)
class NormalPredictorGuideModel(torch.nn.Module):
    output_type = 'continuous_value'

    def __init__(self, 
                 model_cfg:ml_collections.ConfigDict, 
                 general_cfg:ml_collections.ConfigDict, 
                 time_encoder:object, 
                 sigma_noised:float=1.0, 
                 logger:Optional[object]=None) -> None:
        """
        Args:
            model_cfg (ml_collections.ConfigDict): Model specific config dictionary.
            general_cfg (ml_collections.ConfigDict): General (non-model specific) config dictionary.
            time_encoder (object): Time encoder object.
            sigma_noised (float): Sigma of the properties 
                (for which a predictor is setup here) 
                of the fully noised samples (at t=0).
                (Default: 1.0)
            logger (None or object): Optional logger object.
                If None, no logger is used.
                (Default: None)

        """
        # Initialize the parent class
        super().__init__()

        # Assign inputs to class attributes
        self.D                  = general_cfg.data.shape
        self.S                  = general_cfg.data.S
        self.y_guide_name       = model_cfg.y_guide_name
        self.hidden_dims        = model_cfg.hidden_dims
        self.stack_time         = model_cfg.stack_time
        self.p_dropout          = model_cfg.p_dropout
        self.eps                = float(model_cfg.eps)
        self.time_encoder       = time_encoder
        self.logger             = logger
        self.output_layer_dim   = 1
        self.sigma_noised       = sigma_noised
        self.log_sigma_unnoised = torch.nn.Parameter(torch.log(torch.tensor(sigma_noised))) # Initialize equal to log(sigma_noised)
        self.unnormalized       = model_cfg.get("unnormalized", False)  # unnormlaized (bool): If True, return the unnormalized log probabilities for self.log_prob().
        self.display(f"Stack time to x as {self.y_guide_name}-predictor model input: {self.stack_time}")

        # Define input dimension
        self.input_dim = self.D*self.S
        if self.stack_time:
            # If the time should be stacked, add the time 
            # encoding dimension to the input dimension
            self.input_dim += self.time_encoder.dim

        # Define the t encoder
        self.encode_t = lambda t: self.time_encoder(t)

        # Define the model parts
        linear_list       = [torch.nn.Linear(self.input_dim, self.hidden_dims[0])]
        linear_list      += [torch.nn.Linear(self.hidden_dims[layer_id-1], self.hidden_dims[layer_id]) for layer_id in range(1, len(self.hidden_dims))]
        self.linear_list  = torch.nn.ModuleList(linear_list)
        self.linear_last  = torch.nn.Linear(self.hidden_dims[-1], self.output_layer_dim) # Output layer

        # Define an activation function
        self.activation_fn = activation_fn_factory(model_cfg)

        # Define (global) dropout function
        self.dropout_fn = torch.nn.Dropout(p=self.p_dropout, inplace=False)
    
    @property
    def device(self) -> object:
        """
        Return the device the model parameters are on.
        
        Returns:
            (object): The device the model parameters are on.
        
        """
        # Pick the first parameter and return on which device it is on
        return next(self.parameters()).device
    
    @property
    def num_parameters(self) -> int:
        """
        Return the number of model parameters.
        
        Source: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/10

        Return:
            (int): Number of model parameters.
        
        """
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def encode_x(self, x:torch.tensor) -> torch.tensor:
        """
        One-hot encode the input tensor x.

        Args:
            x (torch.tensor): Torch tensor of shape (B, D) holding
                'discrete states' entries, where B is the batch size
                and D is the dimensionality of each batch point.

        Return:
            (torch.tensor): Tensor where the 'discrete states' entries 
                (with cardinality S in each dimension) of x have been 
                one-hot encoded to a tensor of shape (B, D, S)

        """
        return torch.nn.functional.one_hot(x.long(), num_classes=self.S).float()
    
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
            
    def forward(self, 
                batch_data_t:torch.tensor, 
                t:torch.tensor=torch.tensor(1.0), 
                is_x_onehot:bool=False) -> torch.tensor:
        """
        Define forward pass of the model.
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): Probabilities for each component of the encoded property 'y'
                as 2D torch tensor of shape (batch_size, dim[encoded(y)])
        
        """
        # Get xt and t, encode both, and stack them
        xt = batch_data_t['x']
        B = xt.shape[0]
        if t.shape == ():
            # If t is a scalar, expand it to the batch size
            t = t.repeat(B).to(self.device)

        # Differ the cases where x is already encoded or not
        if is_x_onehot and xt.shape[-1] == self.S:
            # xt is already encoded
            xt_enc = xt # (B, D, S)
        else:
            # xt has to be encoded
            xt_enc = self.encode_x(xt) # (B, D, S)

        # Flatten features
        xt_enc = xt_enc.view(-1, self.D*self.S) # (B, D*S)

        # Stack encoded time to h if requested
        if self.stack_time:
            t_enc = self.encode_t(t).view(-1, 1)
            h = torch.cat([xt_enc, t_enc], dim=-1) 
        else:
            h = xt_enc

        # Perform pass through the network
        for layer_id in range(len(self.linear_list)):
            h = self.dropout_fn(h)
            h = self.linear_list[layer_id](h)
            h = self.activation_fn(h)
        
        # Perform pass through last layer
        h = self.dropout_fn(h)
        h = self.linear_last(h) # (B, #classes)

        return h.squeeze()
    
    @property
    def sigma_unnoised(self) -> torch.tensor:
        """
        Return the unnoised sigma based on the model parameter 'self.log_sigma_unnoised'.

        Return:
            (torch.tensor): Unnoised sigma.

        """
        return torch.exp(self.log_sigma_unnoised)

    def get_sigma_t(self, t:torch.tensor) -> torch.tensor:
        """
        Return sigma(t) as a linear interpolation between the
        noised sigma at t=0 and the unnoised sigma at t=1.
        
        Remark: 
        At t=1 we have the data distribution (unnoised)
        and at t=0 we have the noised distribution.
        Thus, interpolate the predictor sigma in a similar way.

        Args:
            t (torch.tensor): Times.

        Return:
            (torch.tensor): Determined sigma(t).
        
        """
        return t*self.sigma_unnoised+(1-t)*self.sigma_noised

    def plot_sigma_t(self) -> object:
        """
        Plot sigma(t) vs. t and return the resulting figure.
        
        Return:
            (object): Matplotlib figure object.
        
        """
        fig = plt.figure()
        t = torch.linspace(0, 1, 1000).to(self.sigma_unnoised.device)
        sigma_t = self.get_sigma_t(t)
        plt.plot(utils.to_numpy(t), utils.to_numpy(sigma_t), 'b-')
        plt.xlabel('t')
        plt.ylabel('sigma(t)')
        plt.xlim([0, 1])
        plt.ylim([0, max(utils.to_numpy(sigma_t))*1.05])
        plt.show()
        return fig

    
    def log_prob(self, 
                 batch_data_t:torch.tensor, 
                 t:torch.tensor, 
                 is_x_onehot:bool=False) -> torch.tensor:
        """
        Return the log probability given the data. 
        
        Args:
            batch_data_t (dict): Dictionary containing batched noised 
                input 'x' (at times t) containing discrete states 
                [shape (B, D)] or one-hot encoded states [shape (B, D, S)]. 
                It can also contain additional quantities such as the
                properties {'x': ..., '<y-property-name>': <batched-property-values}
                where <y-property-name> could be defined in self.y_guide_name.
            t (torch.tensor): (Batched) time as 1D torch tensor of
                shape (B,).
            is_x_onehot (bool): Is the x input already encoded or not?
                (Default: False)

        Return:
            (torch.tensor): (Batched) log-probability for each point in the batch as
                1D torch tensor of shape (batch_size,).
        
        """
        y_data = batch_data_t[self.y_guide_name]

        # Determine the class-probabilities
        y_pred = self.forward(batch_data_t, t, is_x_onehot=is_x_onehot) # Shape (B, #classes)

        if self.unnormalized:
            # Calculate the unnormalized log-probability for each point
            # and each category/feature and then sum over the feature (i.e. the second) axis
            log_prob = -((y_data - y_pred) ** 2)
            return log_prob
        
        # Determine sigma(t) and log(sigma(t))
        sigma_t     = self.get_sigma_t(t).squeeze()+self.eps
        log_sigma_t = torch.log(sigma_t)

        # Calculate the log_prob per point
        square_diff = (y_data.squeeze()-y_pred.squeeze())**2/(2*sigma_t**2)
        # log_prob = -square_diff-log_sigma_t-np.sqrt(2*np.pi) ## TODO: should be log(np.sqrt(2*np.pi)
        log_fixed_prob = -square_diff-log_sigma_t-np.log(np.sqrt(2*np.pi))
        log_prob = log_fixed_prob
        return log_prob
    

def activation_fn_factory(model_cfg:ml_collections.ConfigDict) -> object:
    """
    Factory that returns a torch activation function object
    based on the passed input config dictionary.

    Args:
        model_cfg (ml_collections.ConfigDict): Model specific config
            dictionary that should contain 'model_cfg.activation_fn'
            as entry, which is used to construct the torch activation
            function object.
    Return:
        (object): Torch activation function object.

    """
    # Use a ReLU activation function as default
    if 'activation_fn' in model_cfg:
        activation_fn_name = model_cfg.activation_fn.name
        if 'params' in model_cfg.activation_fn:
            activation_fn_params = model_cfg.activation_fn.params
        else:
            activation_fn_params = None
    else:
        # Use a ReLU activation function as default if 
        # the activation function is not defined in the 
        # model config
        activation_fn_name   = 'ReLU'
        activation_fn_params = None

    # Try to get a handle on the activation function
    try:
        activation_fn_handle = getattr(torch.nn, activation_fn_name)
    except AttributeError:
        err_msg = f"There is no activation function in torch.nn with name '{activation_fn_name}'."
        raise ValueError(err_msg)
    
    # Initialize the activation function with parameters if specified
    if activation_fn_params is None:
        # Initialize without parameters
        return activation_fn_handle()
    else:
        # Initialize with parameters
        return activation_fn_handle(**activation_fn_params)


class OraclePredictor:
    output_type = 'continuous_value'
    
    def __init__(
            self, smiles_encoder: SMILESEncoder, 
            property_name="qed", 
            ignore_invalid: bool = False,
            vina_target: str = "parp1",
            use_gpu:bool=True,
            num_cpu:int=1,
            gpu_thread:int=8000,
            gpu_parallel:bool=True,
            eval_batch_size:int=16,
            base_temp_dir: str = "/tmp/tdc_docking",
            exclude_failure_score: bool = False,
        ):
        self.y_guide_name = property_name
        self.default_property = 0.0
        self.smiles_encoder = smiles_encoder # orchestrator.molecules_data_handler.smiles_encoder
        self.oracle = None
        try:
            self.oracle = tdc.Oracle(name=property_name)
        except Exception as e:
            print(f"Fail to load oracle for {property_name} from tdc: {e}")
        self.ignore_invalid = ignore_invalid
        self.vina_target = vina_target
        self.total_num_query_call = 0

        self.docking_kwargs = {
            'use_gpu': use_gpu,
            'num_cpu': num_cpu,
            'gpu_thread': gpu_thread,
            'gpu_parallel': gpu_parallel,
            'eval_batch_size': eval_batch_size,
            'base_temp_dir': base_temp_dir,
            'exclude_failure_score': exclude_failure_score,
        }
        self.run_time = 0.0
        self.update_num_query_call = True


    def __call__(self, batch_data_t, is_x_onehot=False):
        ### qed, sa, jnk3, gsk3b, drd2, logp, docking_score, etc.
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        smiles_encoded_tensor = batch_data_t['x']
        property_list = []
        smiles_list = []
        # valid_num = 0
        for smiles_encoded in smiles_encoded_tensor:
            smiles = self.smiles_encoder.decode(utils.to_numpy(smiles_encoded))
            ### TODO: may need to test the non-stereochemical washed canonical SMILES (nswcs) of the SMILES
            ## reference:
            # # Get the non-stereochemical washed canonical SMILES (nswcs) of the SMILES
            # # If this throws a Kekulize Exception, continue to next molecule
            # try:
            #     nswcs = cheminf.get_washed_canonical_smiles(smiles, remove_stereochemistry=True)
            # except KekulizeException:
            #     print(f"Cannot kekulize the smiles string:\n{smiles}\n Treating it as invalid.")
            #     continue
            smiles_list.append(smiles)
            # property_value, num_query_call = get_property_value(smiles, property_name=self.y_guide_name, strict=False, oracle=self.oracle, vina_target=self.vina_target, ignore_invalid=self.ignore_invalid)
            # # mol = Chem.MolFromSmiles(smiles)
            # # if mol is not None:
            # #     valid_num += 1
            # property_list.append(property_value)
            # self.total_num_query_call += num_query_call

        property_list, num_query_call = get_property_value_all(
            smiles_list, property_name=self.y_guide_name, strict=False, 
            oracle=self.oracle, vina_target=self.vina_target, ignore_invalid=self.ignore_invalid,
            **self.docking_kwargs
        )
        if self.update_num_query_call:
            self.total_num_query_call += num_query_call
            torch.cuda.synchronize()
            self.run_time += time.perf_counter() - start_time
        else:
            print("Will not update the total num query call and run time.")
        property_tensor = torch.FloatTensor(property_list).to(smiles_encoded_tensor.device)
        return property_tensor

        # mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        # if self.reward_name == "qed":
        #     qeds = [QED.qed(mol) if mol is not None else 0 for mol in mols]
        #     return qeds

        # elif self.reward_name == "sa":
        #     scores = [ (10.0 - sascorer.calculateScore(mol))/9.0 if mol!=None else 0.0 for mol in gen_mols]
        
    def log_prob(self, batch_data_t, is_x_onehot=False):

        y = batch_data_t[self.y_guide_name]
        y_pred = self(batch_data_t, is_x_onehot=is_x_onehot).to(y.device)

        unnormalized_log_prob = -((y - y_pred) ** 2) #/ (2 * self.sigma_noised ** 2)
        return unnormalized_log_prob


if __name__ == '__main__':
    predict_on_x1 = True
    New_class = x0_predictor(predict_on_x1=predict_on_x1)(NormalPredictorGuideModel)