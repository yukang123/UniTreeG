
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
import os
os.chdir(os.path.join(root, 'applications/molecules')) 

## could set cache dir to a place with enough space
os.environ['XDG_CACHE_HOME'] = "/scratch/gpfs/yy1325/.cache"

import argparse
import collections
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from applications.molecules.src import config_handling
import numpy as np
from collections import defaultdict

# Only run as main
if __name__=='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('-c', '--config',                     type=str, required=True, help='[Required] Path to (generation) config file.')
    parser.add_argument('-n', '--num_valid_molecule_samples', type=int, required=True, help='[Required] Number of valid molecules to be sampled.')
    parser.add_argument('-p', '--property_name_value',        type=str, default='',    help='[Optional] Which target property value the molecule generation should be guided to in the form "<property-name>=<property-value>" (for example "num_rings=0"). If this argument is not passed, use unconditional generation.')
    parser.add_argument('-o', '--overrides',                  type=str, default='',    help='[Optional] Which configs (in the config file) to override (pass configuration names and override-values in the format "<config-name-1>=<config-value-1>|<config-name-2>=<config-value-2>"). If this argument is not passed, no configurations will be overriden.')
    parser.add_argument("-r", "--folder_remark",             type=str, default='',    help='[Optional] Remark to add to the folder name.')
    parser.add_argument("-uo", "--use_oracle",            action='store_true', help='[Optional] Use oracle as reward function.')
    parser.add_argument("-df", "--debug_predictor_form", action='store_true', help='[Optional] Use the debug form of the predictor model.')
    parser.add_argument("-dv", "--debug_version", type=str, default='v1', help='[Optional] Use the debug version of the model.')
    parser.add_argument("-dh", "--debug_hyper", type=float, default=1.0, help='[Optional] Use the debug hyperparameter of the model.')
    parser.add_argument("-igi", "--ignore_invalid",            action='store_true', help='[Optional] Ignore invalid samples.')
    parser.add_argument("--make_figs", help="make figures", type=bool, required=False, default=True)
    parser.add_argument("--no_make_figs", dest='make_figs', action='store_false')
    args = parser.parse_args()
    
    # Strip potenial '"' at beginning and end of args.overrides
    args.overrides = args.overrides.strip('"')
    # Load the configs from the passed path to the config file
    try:
        method_tag = os.path.basename(args.config).split("_generation")
        assert len(method_tag) > 1
        method_tag = method_tag[0] + "/"
    except:
        method_tag = ""

    generation_cfg = config_handling.load_cfg_from_yaml_file(args.config)
    # Deepcopy the original cfg
    original_generation_cfg = copy.deepcopy(generation_cfg)
    # Parse the overrides
    overrides = config_handling.parse_overrides(args.overrides)
    # Update the configs with the overrides
    generation_cfg.update(overrides)

    if "predict_on_x1" not in generation_cfg:
        generation_cfg.predict_on_x1 = False #predict_on_x1
   
    from applications.molecules.scripts import utils
    utils.PREDICT_ON_X1 = generation_cfg.predict_on_x1 #predict_on_x1
    # Import custom modules
    from applications.molecules.src import bookkeeping
    from applications.molecules.src import cheminf
    from applications.molecules.src import factory
    from applications.molecules.src import logging
    from applications.molecules.src import managers
    from applications.molecules.src import plotting
    from applications.molecules.src import utils

    # Parse the property name and target property value
    args.property_name_value = args.property_name_value.strip('"')
    if args.property_name_value=='': # Default value if no property value was passed
        property_name                   = 'None'
        target_property_value           = None
        run_folder_name_property_prefix = 'unconditional'
    else: # Target property passed
        property_name, target_property_value = args.property_name_value.split('=')
        if target_property_value=='None' or target_property_value=='':
            target_property_value = np.inf #None
        else:
            target_property_value = float(target_property_value)
            if int(target_property_value)==target_property_value:
                target_property_value = int(target_property_value)

        run_folder_name_property_prefix = args.property_name_value

    # Extract the requested number of unique valid nswcs (uvnswcs) from the arguments
    num_uvnswcs_requested = int(args.num_valid_molecule_samples)

    # Create a folder for the current generation run
    save_location = str(Path(generation_cfg.base_dir,  method_tag + 'generated'))
    run_folder_name = f"{run_folder_name_property_prefix}|n={num_uvnswcs_requested}"
    if args.overrides!='':
        run_folder_name += f"|{args.overrides}"
    predictor_model_remark = f"_use_oracle_{args.use_oracle}"
    if args.use_oracle:
        predictor_model_remark += f"_ignore_invalid_{args.ignore_invalid}"
    outputs_dir = bookkeeping.create_run_folder(save_location, run_folder_name, include_time=False, folder_remark=predictor_model_remark + args.folder_remark)

    # Define a logger
    log_file_path = str(Path(outputs_dir, 'logs'))
    logger = logging.define_logger(log_file_path, file_logging_level='INFO', stream_logging_level='DEBUG')

    # Set the logging level of matplotlib to 'info' (to avoid a plethora of irrelevant matplotlib DEBUG logs)
    plt.set_loglevel('info')

    # Log initial information
    if target_property_value is None:
        logger.info(f"Unconditional generation.")
    else:
        logger.info(f"Guided generation towards target: {property_name}={target_property_value}")
    logger.info(f"Generate until the sampled number of unique valid nswcs is: {num_uvnswcs_requested}")

    # Construct an orchestrator from a (trained) 'run folder' containing the saved model weights 
    # and other meta-information required to construct the models and generate from them.
    # Set some overrised for the train_cfg
    trained_overrides = {
        'make_figs': args.make_figs, # True,
        'save_figs': False,
    }
    config_handling.update_dirs_in_cfg(trained_overrides, outputs_dir=generation_cfg.trained_run_folder_dir)
    while True:
        start_time_stamp = int(time.time())
        base_temp_dir = f'/tmp/yukang/base_tmp_{start_time_stamp}'
        try:
            os.makedirs(base_temp_dir, exist_ok=False)
            break
        except:
            continue
    docking_eval_kwargs = {
        "use_gpu": generation_cfg.get('use_gpu', True),
        "num_cpu": generation_cfg.get('num_cpu', 1),
        "gpu_thread": generation_cfg.get('gpu_thread', 8000),
        "gpu_parallel": generation_cfg.get('gpu_parallel', True),
        "eval_batch_size": generation_cfg.get('eval_batch_size', 16),
        "vina_target": generation_cfg.get('vina_target', 'parp1'),
        "base_temp_dir": base_temp_dir,
        "exclude_failure_score": generation_cfg.get('eval_exclude_failure_score', False),
    }

    # Define an orchestrator from the run folder and overrides
    orchestrator = factory.Orchestrator.from_run_folder(
        run_folder_dir=generation_cfg.trained_run_folder_dir, 
        overrides=trained_overrides, load_data=True, 
        logger=logger, predict_on_x1=generation_cfg.predict_on_x1,
        oracle_property_name=property_name,
        ignore_invalid=args.ignore_invalid,
        **docking_eval_kwargs
    )

    # Load all models
    orchestrator.manager.load_all_models() ## TODO: maybe only load the models that are needed

    # Update the generation configurations without overwriting entries
    config_handling.update_without_overwrite(generation_cfg, orchestrator.cfg)

    # Update the directoris in the generation config file
    config_handling.update_dirs_in_cfg(generation_cfg, str(outputs_dir))

    # Update the 'save_figs' flag
    generation_cfg.save_figs = True

    # Define manager for evaluation with trained model
    eval_manager = managers.DFMManager(generation_cfg, 
                                       denoising_model=orchestrator.manager.denoising_model,
                                       predictor_models_dict=orchestrator.manager.predictor_models_dict)

    # Log the overrides
    logger.info(f"Overrides: {overrides}")

    # Log the configs
    logger.info(f"Overriden config: {generation_cfg}")

    # Save the cfg, original_cfg, and overrides as yaml files in cfg.config_dir
    file_path = str(Path(generation_cfg.configs_dir, 'original_config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, original_generation_cfg.to_dict())
    file_path = str(Path(generation_cfg.configs_dir, 'overrides.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, overrides)
    file_path = str(Path(generation_cfg.configs_dir, 'config.yaml'))
    config_handling.save_dict_to_yaml_file(file_path, generation_cfg.to_dict())

    # Define a writer used to track training
    tensorboard_writer = bookkeeping.setup_tensorboard(generation_cfg.outputs_dir, rank=0)

    # Set a random seed
    utils.set_random_seed(generation_cfg.sampler.seed)

    # Generate until a certain number of unique valid nswcs (uvnswcs) has been sampled 
    sampled_uvnswcs_list = list() # Keep track of the sampled unique valid nswcs (uvnswcs)
    generated_df_list    = list()
    global_start_time    = time.time()
    logger.info(f"Will generate molecules are at least {num_uvnswcs_requested} molecules (i.e. nswcs) have been sampled.")

    total_num_samples = 0
    total_valid_samples = 0
    iter_time_info_dict = defaultdict(list)
    for iteration in range(generation_cfg.sampler.max_iterations):
        # If no property is specified, use unconditional sampling
        orchestrator.manager.models_dict["oracle_predictor_model"].update_num_query_call = True
        if target_property_value is None:
            x_generated, time_info_dict = eval_manager.generate(num_samples=generation_cfg.sampler.batch_size, 
                                                seed=None, # Only use the external seed
                                                stochasticity=generation_cfg.sampler.noise,
                                                dt=generation_cfg.sampler.dt,
                                                batch_size=generation_cfg.sampler.batch_size)
        else:
            # Construct the predictor model name based on the property name
            predictor_model_name = f"{property_name}_predictor_model" if not args.use_oracle else f"oracle_predictor_model"
            logger.info(f"Predictor model name: {predictor_model_name}")

            # Check that the predictor model is valid
            if predictor_model_name not in orchestrator.manager.models_dict:
                err_msg = f"There is no predictor model with name '{predictor_model_name}'. Allowed predictor models are: {list(orchestrator.manager.models_dict.keys())}"
                raise ValueError(err_msg)

            x_generated, time_info_dict = eval_manager.generate(num_samples=generation_cfg.sampler.batch_size,
                                                seed=None, # Only use the external seed
                                                stochasticity=generation_cfg.sampler.noise,
                                                dt=generation_cfg.sampler.dt,
                                                predictor_y_dict={predictor_model_name: target_property_value},
                                                guide_temp=generation_cfg.sampler.guide_temp,
                                                grad_approx=generation_cfg.sampler.grad_approx,
                                                batch_size=generation_cfg.sampler.batch_size, 
                                                debug_predictor_form=args.debug_predictor_form,
                                                debug_version=args.debug_version,
                                                debug_hyper=args.debug_hyper,
                                                )
        
        for key, value in time_info_dict.items():
            iter_time_info_dict[key].append(value)
        orchestrator.manager.models_dict["oracle_predictor_model"].update_num_query_call = False
        # Analyze the generated x
        generated_smiles_list = [orchestrator.molecules_data_handler.smiles_encoder.decode(utils.to_numpy(smiles_encoded)) for smiles_encoded in x_generated]
        analysis_dict = utils.analyze_generated_smiles(generated_smiles_list, 
                                                       orchestrator.molecules_data_handler.subset_df_dict['train']['nswcs'],
                                                       pad_token=orchestrator.molecules_data_handler.pad_token,
                                                       logger=logger)

        # Construct a table with the generated molecules
        iter_dict = collections.defaultdict(list)

        for x, smiles in zip(x_generated, generated_smiles_list):    
            # Get the validity
            valid = (smiles in analysis_dict['unique_valid_gen_smiles_list'])

            # If the smiles is not valid, continue to next smiles
            if valid==False:
                continue

            # Determine the nswcs
            # Remark: This is only possible for valid molecules (hence do this after the validation filter above)!
            nswcs = analysis_dict['smiles_to_nswcs_map'][smiles]

            # Append to corresponding lists
            iter_dict['smiles'].append(smiles)
            iter_dict['valid'].append(valid)
            iter_dict['nswcs'].append(nswcs)
            # If a property has been defined, determine the property value(s) of the generated molecule 
            if property_name!='None':
                # Determine the ground truth (using RDKit) property value
                if property_name == "vina_docking":
                    # logger.info("Repeat 48 times to get the mean docking score for each sample")
                    ground_truth_property_value, _ = cheminf.get_docking_score_repeat_mean(smiles, ignore_invalid=False, **docking_eval_kwargs) #  vina_target=generation_cfg.vina_target,           
                else:
                    ground_truth_property_value, _ = cheminf.get_property_value(smiles, property_name=property_name, strict=True, ignore_invalid=False, **docking_eval_kwargs) #  vina_target=generation_cfg.vina_target, 

                predictor_model_name = f"{property_name}_predictor_model" if not args.use_oracle else f"oracle_predictor_model"
                logger.info(f"Predictor model name: {predictor_model_name}")

                # Check that the predictor model is valid
                if predictor_model_name not in orchestrator.manager.models_dict:
                    err_msg = f"There is no predictor model with name '{predictor_model_name}'. Allowed predictor models are: {list(orchestrator.manager.models_dict.keys())}"
                    raise ValueError(err_msg)
                
                # Predict the property value using the corresponding predictor model
                predicted_property_value = orchestrator.manager.predict_property(predictor_model_name, x=x, t=1, return_probs=False)

                # Append to corresponding lists
                if target_property_value is None:
                    iter_dict[f"target_{property_name}"].append('None')
                else:
                    iter_dict[f"target_{property_name}"].append(target_property_value)
                iter_dict[f"predicted_{property_name}"].append(predicted_property_value)
                iter_dict[f"ground_truth_{property_name}"].append(ground_truth_property_value)

        # Transform the dictionary of lists to a pandas.DataFrame
        iter_df = pd.DataFrame(iter_dict)
        
        # Make the iter_dict a pandas DataFrame and append it to the corresponding list
        generated_df_list.append(pd.DataFrame(iter_dict))

        # Update the list of unique valid nswcs (uvnswcs)
        if 0<len(iter_df): # If there were no molecules in this iteration, we cannot update
            filtered_df = iter_df[iter_df['valid']==True]
            sampled_uvnswcs_list += list(set(filtered_df['nswcs']))
            sampled_uvnswcs_list = list(set(sampled_uvnswcs_list))

        # Determine the number of sampled unique valid nswcs (uvnswcs)
        num_sampled_uvnswcs = len(sampled_uvnswcs_list)
        total_num_samples += len(x_generated)

        logger.info(f"[{iteration}] Number of already sampled unique valid nswcs: {num_sampled_uvnswcs} (Duration since start: {(time.time()-global_start_time)/60:.2f}min)")
        logger.info('-'*100)
        tensorboard_writer.add_scalar(f"{property_name}/Num-sampled-molecules", num_sampled_uvnswcs, iteration)

        # If the number of unique valid generated nswcs exceeds the requestes number, 
        # halt generation
        if num_uvnswcs_requested<=num_sampled_uvnswcs:
            break

    # Log the final information
    logger.info(f"Generated at least {num_uvnswcs_requested} valid molecules. Duration: {(time.time()-global_start_time)/60:.2f}min")
    if len(iter_time_info_dict)>0:
        logger.info(f"Detailed Time information:")
        for time_key, time_list in iter_time_info_dict.items():
            if time_key == "batch_size" or time_key == "valid_steps":
                postfix = ""
            else:
                postfix = "s"
            logger.info(f"[Average over Batch] {time_key}: {np.mean(time_list):.8f}{postfix}")

    # Stack the DataFrames in the list 'generated_df_list' to obtain one big DataFrame
    generated_df = pd.concat(generated_df_list)
    file_path = str(Path(generation_cfg.outputs_dir, 'full_samples_table.tsv')) 
    generated_df.to_csv(file_path, index=False, sep='\t')


    filtered_df = generated_df[generated_df['valid']==True]
    total_valid_samples = len(filtered_df)
    unique_valid_samples = list(set(filtered_df['smiles'])) 
    total_unique_valid_samples = len(unique_valid_samples)


    mae_list = []
    gt_value_list = []
    unique_df = []
    for smile in unique_valid_samples: #[:args.num_valid_molecule_samples]: 
        ## COULD ONLY SELECT THE FIRST args.num_valid_molecule_samples SAMPLES TO MAKE SURE THE SAMPLE SIZES ARE EXACTLY THE SAME ACROSS DIFFERENT SETTINGS
        sub_df = filtered_df[filtered_df["smiles"]==smile]
        unique_df.append(sub_df.iloc[0]) ## only select the first one
        if target_property_value is not None:
            gt_values = sub_df[f"ground_truth_{property_name}"].values
            mae = np.abs(np.mean(gt_values) - target_property_value)
            gt_value_list.append(np.mean(gt_values))
            mae_list.append(mae)
        
        avg_gt_value = np.mean(gt_value_list)
        avg_mae = np.mean(mae_list)
        std_mae = np.std(mae_list)
    
    unique_df = pd.DataFrame(unique_df)
    file_path = str(Path(generation_cfg.outputs_dir, 'unique_valid_samples_table.tsv')) 
    unique_df.to_csv(file_path, index=False, sep='\t')
    

    logger.info(f"Total Number of generated nswcs: {total_num_samples}")
    logger.info(f"Total Number of valid generated nswcs: {total_valid_samples} ({100*total_valid_samples/total_num_samples:.2f}%)")
    logger.info(f"Total Number of unique valid generated nswcs: {total_unique_valid_samples} ({100*total_unique_valid_samples/total_num_samples:.2f}%)")
    oracle_num_query_call = orchestrator.manager.models_dict["oracle_predictor_model"].total_num_query_call if "oracle_predictor_model" in orchestrator.manager.models_dict else 0
    oracle_runtime = orchestrator.manager.models_dict["oracle_predictor_model"].run_time if "oracle_predictor_model" in orchestrator.manager.models_dict else 0.0
    logger.info(f"Total Number of oracle query calls: {oracle_num_query_call}")
    logger.info(f"Total Runtime of oracle query calls: {oracle_runtime:.2f}s")
    logger.info(f"Runtime Per Oracle Query Call: {oracle_runtime/(oracle_num_query_call+1e-9):.2f}s")
    if target_property_value is not None:
        logger.info(f"[unique valid generation ({len(mae_list)})] MAE compared with target {target_property_value}: Average {avg_mae:.2f} | Std {std_mae:.2f}")
        logger.info(f"[unique valid generation ({len(gt_value_list)})] generated property values: Average {avg_gt_value:.2f} | Std {np.std(gt_value_list):.2f}")

    # Only make a plot if there are any unique valid generated smiles
    if len(unique_valid_samples)>0 and generation_cfg.make_figs:

        property_names = ['num_tokens', 'logp', 'num_rings', 'num_heavy_atoms', "qed", "sa", "drd2", "vina_docking"]
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        if target_property_value is None:
            guide_temp_label = None
        else:
            guide_temp_label = generation_cfg.sampler.guide_temp

        logger.info(f"=====================Property Evaluation ({len(unique_valid_samples)})=====================")
        plt.suptitle(f"Target {property_name}: {target_property_value} | Stochasticity: {generation_cfg.sampler.noise} | T: {guide_temp_label}")
        for index, property_name in enumerate(property_names):
            index1 = index%3 #2
            index2 = (index-index1)//3 #2
            ax = axs[index1, index2]
            gen_property_values = plotting.plot_gen_vs_train_distribution(property_name, 
                                                    orchestrator.molecules_data_handler.subset_df_dict['train'], 
                                                    unique_valid_samples,
                                                    ax=ax,
                                                    **docking_eval_kwargs)
            assert len(gen_property_values) == len(unique_valid_samples), f"Length of gen_property_values ({len(gen_property_values)}) does not match the number of unique valid generated smiles ({len(unique_valid_samples)})"
            logger.info(f"[{property_name}] Average {np.mean(gen_property_values):.2f} | Std {np.std(gen_property_values):.2f}")
            
        # Save the figure
        if generation_cfg.save_figs and generation_cfg.figs_save_dir is not None:
            file_path = str(Path(generation_cfg.figs_save_dir, f"Visualization_samples.png"))
            fig.savefig(file_path)
                        