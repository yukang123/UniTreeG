# Import public modules
import ml_collections
import os
import torch
from pathlib import Path
from typing import Optional

# Import custom modules
from . import config_handling
from . import data_handling
from . import managers
from . import models
from . import preprocessing
from . import utils

class Orchestrator(object):
    """
    Orchestrator used to orchestrate various processes.
    """
    def __init__(self, 
                cfg:ml_collections.ConfigDict, 
                logger:Optional[object]=None, 
                load_data:Optional[bool]=None,
                oracle_property_name:Optional[str]='qed', ## check cheminf.get_property_value()
                debug: bool=False,
                vina_target: str = "parp1",  
                ignore_invalid: bool = False,  # If True, ignore invalid SMILES and return None for them.
                **docking_eval_kwargs
                #  remark: Optional[str]='',
                 ) -> None:
        """
        Args:
            cfg (ml_collections.ConfigDict): Config dictionary.
            logger (None or object): Optional logger object.
                (Default: None)
            load_data (None or bool): Optional flag to request data loading.
                If None, use the boolean flag in cfg (i.e., cfg.load_data).
                (Default: None)

        """
        # If load_data was not passed, use the corresponding entry in the configs
        if load_data is None:
            load_data = cfg.load_data

        ################################################################################
        ### Step 1: Load and preprocess the data
        ################################################################################
        if load_data:
            # Load the dataset
            dataset_remark = cfg.data.get('remark', "")
            dataset_df = preprocessing.load_preprocessed_dataset(cfg.data.which_dataset, 
                                                                 base_dir=cfg.base_dir, 
                                                                 logger=logger, remark=dataset_remark)

            # Initialize the molecules data handler
            molecules_data_handler = data_handling.MoleculesDataHandler(cfg.data.preprocessing, 
                                                                        dataset_df.copy(deep=True), 
                                                                        make_figs=cfg.make_figs, 
                                                                        save_figs=cfg.save_figs, 
                                                                        figs_save_dir=cfg.figs_save_dir, 
                                                                        logger=logger)

            if debug: ## [DEBUG]
                self.molecules_data_handler = molecules_data_handler
                return
            # Map the dataset to the training device
            molecules_data_handler.to(cfg.device)

            # Update certain configs using the molecules data handler
            cfg.data.shape     = molecules_data_handler.max_num_tokens
            cfg.data.S         = molecules_data_handler.token_alphabet_size + 1 # Add 1 masked state as additional token
            cfg.data.pad_index = molecules_data_handler.pad_token_index
            cfg.data.train_property_sigma_dict  = molecules_data_handler.train_property_sigma_dict
            cfg.data.train_num_tokens_freq_dict = molecules_data_handler.train_num_tokens_freq_dict

        else:
            # Check that configs set by the molecule handler are not None (i.e. loaded from some config file)
            if (cfg.data.get('shape', None) is None) or (cfg.data.shape is None):
                err_msg = f"As the data is not loaded, 'cfg.data.shape' must be defined in the config file, but it was not."
                raise ValueError(err_msg)
            if (cfg.data.get('S', None) is None) or (cfg.data.S is None):
                err_msg = f"As the data is not loaded, 'cfg.data.S' must be defined in the config file, but it was not."
                raise ValueError(err_msg)
            if (cfg.data.get('pad_index', None) is None) or (cfg.data.pad_index is None):
                err_msg = f"As the data is not loaded, 'cfg.data.pad_index' must be defined in the config file, but it was not."
                raise ValueError(err_msg)
            if (cfg.data.get('train_property_sigma_dict', None) is None) or (cfg.data.train_property_sigma_dict is None):
                err_msg = f"As the data is not loaded, 'cfg.data.train_property_sigma_dict' must be defined in the config file, but it was not."
                raise ValueError(err_msg)
            if (cfg.data.get('train_num_tokens_freq_dict', None) is None) or (cfg.data.train_num_tokens_freq_dict is None):
                err_msg = f"As the data is not loaded, 'cfg.data.train_num_tokens_freq_dict' must be defined in the config file, but it was not."
                raise ValueError(err_msg)

        # Construct dataloaders in case the data has been loaded
        dataloader_dict = dict()
        if load_data:
            # Loop over all datasets
            for set_name, set_torch_dataset in molecules_data_handler.torch_dataset_dict.items():
                if 'train' in set_name:
                    # For any 'train' set (this can also be a property specific one)
                    batch_size = cfg.data.dataloaders.train.batch_size
                    shuffle    = True
                else:
                    # For any 'validation' set
                    batch_size = cfg.data.dataloaders.train.batch_size
                    shuffle    = True

                dataloader_dict[set_name] = torch.utils.data.DataLoader(set_torch_dataset,
                                                                        batch_size=batch_size,
                                                                        shuffle=shuffle, 
                                                                        num_workers=0)

        # Define some time encoder to be used across models
        time_encoder = DummyTimeEncoder()

        ################################################################################
        ### Step 2: Define models and associated optimizers
        ################################################################################
        ## Denoising model
        # Define the model
        utils.set_random_seed(cfg.denoising_model.init_seed)
        denoising_model = models.DenoisingModel(cfg, time_encoder=time_encoder, logger=logger)
        # Define the optimizer
        if load_data:
            optimizer_handle    = getattr(torch.optim, cfg.training.denoising_model.optimizer)
            denoising_optimizer = optimizer_handle(denoising_model.parameters(), lr=cfg.training.denoising_model.lr)
        else:
            denoising_optimizer = None

        ## Number of rings predictor model
        # Define the model
        # Remark: The number of categories for 'number of rings' is the maximal number of rings
        #         plus 1 because there are molecules with zero rings.
        utils.set_random_seed(cfg.num_rings_predictor_model.init_seed)
        if cfg.num_rings_predictor_model.type=='categorical':
            if logger is None:
                logger.info("Using a categorical predictor-guide model for 'num_rings'.")
            else:
                print("Using a categorical predictor-guide model for 'num_rings'.")
            num_rings_predictor_model = models.CategoricalPredictorGuideModel(num_categories=molecules_data_handler.max_num_rings_train+1, 
                                                                              model_cfg=cfg.num_rings_predictor_model, 
                                                                              general_cfg=cfg, 
                                                                              time_encoder=time_encoder,
                                                                              logger=logger)
        elif cfg.num_rings_predictor_model.type=='ordinal':
            if logger is None:
                print("Using an ordinal predictor-guide model for 'num_rings'.")
            else:
                logger.info("Using an ordinal predictor-guide model for 'num_rings'.")
            num_rings_predictor_model = models.OrdinalPredictorGuideModel(num_ordinals=molecules_data_handler.max_num_rings_train+1,
                                                                          model_cfg=cfg.num_rings_predictor_model, 
                                                                          general_cfg=cfg, 
                                                                          time_encoder=time_encoder,
                                                                          sigma_noised=cfg.data.train_property_sigma_dict['num_rings'],
                                                                          logger=logger)
        elif cfg.num_rings_predictor_model.type=='normal':
            if logger is None:
                print("Using a normal predictor-guide model for 'num_rings'.")
            else:
                logger.info("Using a normal predictor-guide model for 'num_rings'.")
            num_rings_predictor_model = models.NormalPredictorGuideModel(model_cfg=cfg.num_rings_predictor_model, 
                                                                         general_cfg=cfg, 
                                                                         time_encoder=time_encoder,
                                                                         sigma_noised=cfg.data.train_property_sigma_dict['num_rings'],
                                                                         logger=logger)
        else:
            raise ValueError(f"train_cfg.num_rings_predictor_model.type must be either 'categorical' or 'ordinal'.")
        # Define the optimizer
        if load_data:
            optimizer_handle              = getattr(torch.optim, cfg.training.num_rings_predictor_model.optimizer)
            num_rings_predictor_optimizer = optimizer_handle(num_rings_predictor_model.parameters(), 
                                                             lr=cfg.training.num_rings_predictor_model.lr)
        else:
            num_rings_predictor_optimizer = None


        ## Logp predictor model
        # Define the model
        utils.set_random_seed(cfg.logp_predictor_model.init_seed)
        logp_predictor_model = models.NormalPredictorGuideModel(model_cfg=cfg.logp_predictor_model, 
                                                                general_cfg=cfg, 
                                                                time_encoder=time_encoder,
                                                                sigma_noised=cfg.data.train_property_sigma_dict['logp'],
                                                                logger=logger)
        # Define the optimizer
        if load_data:
            optimizer_handle         = getattr(torch.optim, cfg.training.logp_predictor_model.optimizer)
            logp_predictor_optimizer = optimizer_handle(logp_predictor_model.parameters(), 
                                                        lr=cfg.training.logp_predictor_model.lr)
        else:
            logp_predictor_optimizer = None

        ## number of heavy atoms predictor model
        # Define the model
        utils.set_random_seed(cfg.logp_predictor_model.init_seed)
        num_heavy_atoms_predictor_model = models.NormalPredictorGuideModel(model_cfg=cfg.num_heavy_atoms_predictor_model, 
                                                                           general_cfg=cfg, 
                                                                           time_encoder=time_encoder,
                                                                           sigma_noised=cfg.data.train_property_sigma_dict['num_heavy_atoms'],
                                                                           logger=logger)
        # Define the optimizer
        if load_data:
            optimizer_handle                    = getattr(torch.optim, cfg.training.num_heavy_atoms_predictor_model.optimizer)
            num_heavy_atoms_predictor_optimizer = optimizer_handle(num_heavy_atoms_predictor_model.parameters(), 
                                                                   lr=cfg.training.num_heavy_atoms_predictor_model.lr)
        else:
            num_heavy_atoms_predictor_optimizer = None

        ## Other models
        # Define the model
        oracle_predictor_model = models.OraclePredictor(
            smiles_encoder = molecules_data_handler.smiles_encoder,
            property_name = oracle_property_name, 
            ignore_invalid  = ignore_invalid,
            vina_target = vina_target,
            **docking_eval_kwargs
        )
        ## Define the predictor models

        # predictor_models_dict = {
        #     'num_rings_predictor_model': {
        #         'model': num_rings_predictor_model,
        #         'optimizer': num_rings_predictor_optimizer, 
        #     },
        #     'logp_predictor_model': {
        #         'model': logp_predictor_model,
        #         'optimizer': logp_predictor_optimizer, 
        #     },
        #     'num_heavy_atoms_predictor_model': {
        #         'model': num_heavy_atoms_predictor_model,
        #         'optimizer': num_heavy_atoms_predictor_optimizer, 
        #     },
        #     'oracle_predictor_model': {
        #         'model': oracle_predictor_model,
        #         'optimizer': None, 
        #     }
        # }

        predictor_models_dict = {
            'oracle_predictor_model': {
                'model': oracle_predictor_model,
                'optimizer': None, 
            }
        }
        
        continuous_property_names = getattr(cfg, 'continuous_property_names', [oracle_property_name])
        for continuous_property_name in continuous_property_names:
            if continuous_property_name in ["logp", "num_heavy_atoms", "num_rings"]:
                # Skip the logp and num_heavy_atoms properties as they are already defined
                if continuous_property_name == "logp":
                    predictor_models_dict["logp_predictor_model"] = {
                        'model': logp_predictor_model,
                        'optimizer': logp_predictor_optimizer, 
                    }
                elif continuous_property_name == "num_heavy_atoms":
                    predictor_models_dict["num_heavy_atoms_predictor_model"] = {
                        'model': num_heavy_atoms_predictor_model,
                        'optimizer': num_heavy_atoms_predictor_optimizer, 
                    }
                elif continuous_property_name == "num_rings":
                    predictor_models_dict["num_rings_predictor_model"] = {
                        'model': num_rings_predictor_model,
                        'optimizer': num_rings_predictor_optimizer, 
                    }
                continue
            
            continuous_predictor_cfg =  getattr(cfg, f"{continuous_property_name}_predictor_model", None) #logp_predictor_model
            if continuous_predictor_cfg is None:
                print(f"[Warning]: cfg.{continuous_property_name}_predictor_model is None. Skipping this model.")
                continue
            # assert continuous_predictor_cfg is not None, f"cfg.{continuous_property_name}_predictor_model must be defined in the config file."
            utils.set_random_seed(continuous_predictor_cfg.init_seed)
            # continuous_predictor_cfg.y_guide_name = oracle_property_name
            sigma_noised = cfg.data.train_property_sigma_dict[continuous_property_name] #['logp']
            continuous_predictor_model = models.NormalPredictorGuideModel(model_cfg=continuous_predictor_cfg, 
                                                                    general_cfg=cfg, 
                                                                    time_encoder=time_encoder,
                                                                    sigma_noised=sigma_noised,
                                                                    logger=logger)
            # Define the optimizer
            if load_data:
                train_model_config = getattr(cfg.training, f"{continuous_property_name}_predictor_model")
                assert train_model_config is not None, f"cfg.training.{continuous_property_name}_predictor_model must be defined in the config file."
                optimizer_handle         = getattr(torch.optim, train_model_config.optimizer)
                continuous_predictor_optimizer = optimizer_handle(continuous_predictor_model.parameters(), 
                                                            lr=train_model_config.lr)
            else:
                continuous_predictor_optimizer = None
            
            predictor_models_dict[f"{continuous_property_name}_predictor_model"] = {
                'model': continuous_predictor_model,
                'optimizer': continuous_predictor_optimizer, 
            }
        print("Predictor Modes:", list(predictor_models_dict.keys()))
        ## Define manager
        manager = managers.DFMManager(cfg, 
                                               denoising_model=denoising_model, 
                                               denoising_optimizer=denoising_optimizer, 
                                               predictor_models_dict=predictor_models_dict,
                                               logger=logger)
        
        ## Assign to class attributes
        self.cfg                    = cfg
        self.manager                = manager
        self.raw_dataset_df         = dataset_df
        self.molecules_data_handler = molecules_data_handler
        self.dataloader_dict        = dataloader_dict

        if logger is None:
            print('')
        else:
            logger.info('')

    @classmethod
    def from_run_folder(cls, 
                        run_folder_dir:str, 
                        overrides:dict={}, 
                        predict_on_x1:bool=False,
                        **kwargs) -> object:
        """
        Initilize an orchestrator from the path to an experimental run folder. 

        Args:
            run_folder_dir (str): Path to the run-folder.
            overrides (dict): Optional overrides for the configurations.
                (Default: {} <=> i.e., no overrides)
        
        Return:
            (object): Orchestrator object created from the run folder.
        
        """
        # Check that the run folder exists
        if os.path.isdir(run_folder_dir)==False:
            err_msg = f"No 'run-folder' found in: {run_folder_dir}"
            raise FileNotFoundError(err_msg)
        
        # Construct the path to the config file located in the run folder and load the configs
        config_file_path = str(Path(run_folder_dir, 'configs', 'config.yaml'))
        cfg = config_handling.load_cfg_from_yaml_file(config_file_path)

        # Update the configs with the overrides
        cfg.update(overrides)

        if cfg.get("predict_on_x1", False) != predict_on_x1: ## TODO: Check if this is the correct way to handle this
            print("[Warning] The 'predict_on_x1' flag in the training configs does not match the one in generation config.")

        # Construct a class instance based on the configs and return it
        return cls(cfg, **kwargs)

class DummyTimeEncoder(object):
    """
    Simplest time encoder that can be used as template for more complex ones.
    """
    def __init__(self) -> None:
        # Encode each time point of the batch by itself (i.e., 1D time encoding).
        self.dim = 1

    def __call__(self, t:torch.tensor) -> torch.tensor:
        """
        Args:
            t (torch.tensor): Time as 1D torch tensor of shape (B,).
        
        Return:
            (torch.tensor): Time as 2D torch tensor of shape (B, 1).
        """
        return t.view(-1, 1)