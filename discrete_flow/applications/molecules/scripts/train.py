import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
import os
os.chdir(os.path.join(root, 'applications/molecules'))

# Import public modules
import argparse
import copy
import matplotlib.pyplot as plt
from pathlib import Path

# Only run as main
if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('-c', '--config',    type=str, required=True, help='[Required] Path to (training) config file.')
    parser.add_argument('-m', '--model',     type=str, required=True, help='[Required] Which model to train.\nUse "all" to train all models. Use "all_predictors" to train all predictor models (but not the denoising model). Use <model-name> (e.g. "denoising_model", "num_rings_predictor_model", "logp_predictor_model", or "num_heavy_atoms_predictor_model") to train a specific model.')
    parser.add_argument('-o', '--overrides', type=str, default='',    help='[Optional] Which configs (in the config file) to override (pass configuration names and override-values in the format "<config-name-1>=<config-value-1>|<config-name-2>=<config-value-2>"). If this argument is not specified, no configurations will be overriden.')
    parser.add_argument('-r', '--remark',    type=str, default='',    help='[Optional] Remark to add to the run folder name. This is useful to distinguish between different runs with the same config and overrides. The remark will be added to the end of the run folder name.')
    args = parser.parse_args()

    # Import custom modules
    from applications.molecules.src import config_handling
    # Load the configs from the passed path to the config file
    cfg = config_handling.load_cfg_from_yaml_file(args.config)

    # Deepcopy the original cfg
    original_cfg = copy.deepcopy(cfg)

    # Strip potenial '"' at beginning and end of args.overrides
    args.overrides = args.overrides.strip('"')

    # Parse the overrides
    overrides = config_handling.parse_overrides(args.overrides)

    # Update the configs with the overrides
    cfg.update(overrides)

    if "predict_on_x1" not in cfg:
        cfg.predict_on_x1 = False

    # from src import fm_utils
    # fm_utils.PREDICT_ON_X1 = cfg.predict_on_x1
    from applications.molecules.scripts import utils
    utils.PREDICT_ON_X1 = cfg.predict_on_x1

    from applications.molecules.src import bookkeeping
    from applications.molecules.src import factory
    from applications.molecules.src import logging
    from applications.molecules.src import model_training
    from applications.molecules.src import plotting


    # Create a folder for the current training run
    save_location = str(Path(cfg.base_dir, 'trained'))
    if args.overrides=='':
        run_folder_name = 'no_overrides'
    else:
        run_folder_name = args.overrides
    run_folder_name += args.remark
    outputs_dir = bookkeeping.create_run_folder(save_location, run_folder_name, include_time=False)
    config_handling.update_dirs_in_cfg(cfg, str(outputs_dir))

    # Define a logger
    log_file_path = str(Path(cfg.outputs_dir, 'logs'))
    logger = logging.define_logger(log_file_path, file_logging_level='INFO', stream_logging_level='DEBUG')

    # Set the logging level of matplotlib to 'info' (to avoid a plethora of irrelevant matplotlib DEBUG logs)
    plt.set_loglevel('info')

    # Log the overrides
    logger.info(f"Overrides: {overrides}")

    # Generate orchestrator from cfg
    # Remark: This will update cfg
    orchestrator = factory.Orchestrator(cfg, logger=logger)

    # Log the configs
    logger.info(f"Overriden config: {cfg}")

    # Save the cfg, original_cfg, and overrides as yaml files in cfg.config_dir
    file_path = str(Path(cfg.configs_dir, 'original_config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, original_cfg.to_dict())
    file_path = str(Path(cfg.configs_dir, 'overrides.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, overrides)
    file_path = str(Path(cfg.configs_dir, 'config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, cfg.to_dict())

    # Define a writer used to track training
    tensorboard_writer = bookkeeping.setup_tensorboard(cfg.outputs_dir, rank=0)

    # Determine the list of models to be trained
    if args.model=='all':
        # Train all models
        train_models = list(orchestrator.manager.models_dict.keys())
    elif args.model=='all_predictors':
        # Train all predictor models
        train_models = list(orchestrator.manager.predictor_models_dict.keys())
    elif args.model in orchestrator.manager.models_dict:
        train_models = [args.model]
    else:
        err_msg = f"Unknown model name. The passed 'model' must be either 'all' (train all models) or one of the defined models, which are: {list(orchestrator.manager.models_dict.keys())}"
        raise ValueError(err_msg)

    # Log the models to be trained
    logger.info(f"Models to be trained: {train_models}")

    # Loop over to be trained models
    for model_name in train_models:
        if model_name=='denoising_model':
            # Denoising model
            train_dataloader = orchestrator.dataloader_dict['train']
        else:
            # Property models
            y_guide_name = orchestrator.manager.models_dict[model_name].y_guide_name
            property_set_name = f"train_{y_guide_name}"
            if property_set_name in orchestrator.dataloader_dict:
                logger.info(f"Using the property-model specific dataloader of the '{property_set_name}' set.")
                train_dataloader = orchestrator.dataloader_dict[property_set_name]
            else:
                logger.info(f"Using the dataloader of the 'train' set.")
                train_dataloader = orchestrator.dataloader_dict['train']

        # Train the model
        model_training.train_model(orchestrator.manager, 
                                   train_dataloader, 
                                   which_model=model_name, 
                                   num_epochs=cfg.training[model_name].num_epochs,
                                   validation_dataloader=orchestrator.dataloader_dict['validation'], 
                                   random_seed=cfg.training[model_name].seed,
                                   plot_training_curves=cfg.make_figs,
                                   figs_save_dir=cfg.figs_save_dir,
                                   tensorboard_writer=tensorboard_writer,
                                   logger=logger)

        # Save the trained model (allowing overwriting)
        orchestrator.manager.save_model(model_name, overwrite=True)

        # Make some figures in case the model is a predictor model (i.e. not the denoising model)
        if model_name!='denoising_model':
            if cfg.make_figs:
                # Sigma(t) plot
                fig = orchestrator.manager.models_dict[model_name].plot_sigma_t()
                if cfg.save_figs and cfg.figs_save_dir is not None:
                    file_path = str(Path(cfg.figs_save_dir, f"log_sigma_t_{model_name}.png"))
                    fig.savefig(file_path)

                # Correlation model to gt on train set
                fig = plotting.make_correlation_plot(model_name, 
                                                       orchestrator,
                                                       set_name=property_set_name,
                                                       t_eval=1, 
                                                       seed=42)
                if cfg.save_figs and cfg.figs_save_dir is not None:
                    file_path = str(Path(cfg.figs_save_dir, f"correlation_plot_train_{model_name}.png"))
                    fig.savefig(file_path)

                # Correlation model to gt on validation set
                fig = plotting.make_correlation_plot(model_name, 
                                                       orchestrator,
                                                       set_name='validation',
                                                       t_eval=1, 
                                                       seed=42)
                if cfg.save_figs and cfg.figs_save_dir is not None:
                    file_path = str(Path(cfg.figs_save_dir, f"correlation_plot_validation_{model_name}.png"))
                    fig.savefig(file_path)
        
        if 1<len(train_models):
            logger.info('-'*100)


