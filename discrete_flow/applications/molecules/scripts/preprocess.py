# Import custom modules
from applications.molecules.src import preprocessing

# Only run as main
if __name__=='__main__':
    # Define the base directory argument for the following functions
    # base_dir = '.'
    base_dir = './applications/molecules'
    # Determine the unique non-stereochemical washed canonical SMILES (nswcs) strings for each of SMILES 
    # appearing in the raw QMugs dataset, and save them as (one-column) table 
    # `/applications/molecules/data/preprocessed_data/qmugs_unique_nswcs_df.tsv`.
    # print('Re-creating the list of unique non-stereochemical washed canonical SMILES (nswcs) and saving it:')
    # preprocessing.create_unique_nswcs_df(which_dataset='qmugs', 
    #                                     base_dir=base_dir, 
    #                                     logger=None)
    # print('\n')

    # Determine molecular properties (#tokens, #rings, #heavy_atoms, logP, molecular-weight) using RDKit for each of the unique nswcs strings and save the
    # resulting table (nswcs string and corresponding properties per row) as `/applications/molecules/data/preprocessed_data/qmugs_preprocessed_dataset.tsv`.
    print('Creating the preprocessed dataset for all unique nswcs:')
    preprocessing.create_preprocessed_dataset(which_dataset='qmugs', 
                                              base_dir=base_dir, 
                                              logger=None, remark="_updated")
