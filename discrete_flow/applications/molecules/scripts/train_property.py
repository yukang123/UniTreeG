import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
import os
os.chdir(os.path.join(root, 'applications/molecules'))

from applications.molecules.src import data_handling, preprocessing, config_handling, logging, plotting
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
run_folder_dir = './trained/2024-12-23/no_overrides'
config_file_path = str(Path(run_folder_dir, 'configs', 'config.yaml'))
cfg = config_handling.load_cfg_from_yaml_file(config_file_path)


# # Define an orchestrator from the run folder and overrides
# orchestrator = factory.Orchestrator.from_run_folder(
#     run_folder_dir=generation_cfg.trained_run_folder_dir, 
#     overrides=trained_overrides, load_data=True, 
#     logger=logger, predict_on_x1=generation_cfg.predict_on_x1,
#     oracle_property_name=property_name,
#     )

# log_file_path = str(Path(outputs_dir, 'logs'))
# logger = logging.define_logger(log_file_path, file_logging_level='INFO', stream_logging_level='DEBUG')


logger = None
# Load the dataset
dataset_df = preprocessing.load_preprocessed_dataset(cfg.data.which_dataset, 
                                                        base_dir=cfg.base_dir, 
                                                        logger=logger)

# Initialize the molecules data handler
molecules_data_handler = data_handling.MoleculesDataHandler(cfg.data.preprocessing, 
                                                            dataset_df.copy(deep=True), 
                                                            make_figs=cfg.make_figs, 
                                                            save_figs=cfg.save_figs, 
                                                            figs_save_dir=cfg.figs_save_dir, 
                                                            logger=logger)

data_df = molecules_data_handler.subset_df_dict['train']


property_names = ['num_tokens', 'logp', 'num_rings', 'num_heavy_atoms', "qed", "sa", "drd2", "gsk3b"]
property_names = ["drd2", "gsk3b", 'num_tokens', 'logp', 'num_rings', 'num_heavy_atoms', "qed", "sa"]

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
plt.suptitle(f"Training set distribution of properties", fontsize=16)
unique_valid_gen_smiles_list = []
for index, property_name in enumerate(property_names):
    index1 = index%3 #2
    index2 = (index-index1)//3 #2
    ax = axs[index1, index2]
    gen_property_values = plotting.plot_train_distribution(property_name, 
                                            data_df, 
                                            ax=ax)
    print(f"Property: {property_name}, Mean: {np.mean(gen_property_values)}, Std: {np.std(gen_property_values)}")
save_folder = "./data/property_distributions"
os.makedirs(save_folder, exist_ok=True)
# file_path = str(Path(generation_cfg.figs_save_dir, f"Visualization_samples.png"))
fig.savefig(os.path.join(save_folder, f"Visualization_samples.png"), dpi=300, bbox_inches='tight')
            
aa = 1

