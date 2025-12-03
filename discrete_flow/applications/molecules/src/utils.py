# Import public modules
import os
import random
import rdkit
import torch
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pathlib import Path
from rdkit.Chem.AllChem import KekulizeException
from typing import List, Dict, Optional

# PREDICT_ON_X1 = False
# Import custom modules
from . import cheminf
    
def set_random_seed(random_seed:int) -> None:
    """ 
    Set random seed(s) for reproducibility. 
    
    Args:
        random_seed (int): Random seed to be used as basis 
            for the seeds of 'random', 'numpy', and 'torch'
            modules.
    
    """
    # Set random seeds for any modules that potentially use randomness
    random.seed(random_seed)
    np.random.seed(random_seed+1)
    torch.random.manual_seed(random_seed+2)

def to_numpy(x:ArrayLike) -> np.ndarray:
    """
    Map input x to numpy array.
    
    Args:
        x (np.ndarray or torch.tensor): Input to be mapped to numpy array.
    
    Return:
        (np.ndarray): Input x casted to a numpy array.
        
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        err_msg = f"The input must be either a numpy array or a torch tensor, got type {type(x)} instead."
        raise TypeError(err_msg)

def analyze_generated_smiles(gen_smiles_list:List[str], 
                             train_nswcs_list:List[str], 
                             pad_token:str, 
                             logger:Optional[object]=None) -> Dict:
    """
    Analyze the generated SMILES strings, display results, and return the results also as dictionary.

    Args:
        gen_smiles_list (list of strings): List of generated SMILES strings.
        train_nswcs_list (list of strings): List of non-stereochemical washed canonical SMILES (nswcs) strings present in the train set.
        pad_token (str): Pad token.
        logger (None or logging object): Optional logger object.
            (Default: None)

    Return:
        (dict): Dictionary containing different results of the analysis.
    
    """
    # Get a list of unique generated SMILES strings
    unique_gen_smiles_list = list(set(gen_smiles_list))

    invalid_padded_gen_smiles_list     = list()
    unique_valid_gen_smiles_list       = list()
    unique_novel_valid_gen_smiles_list = list()
    unique_novel_valid_gen_ncsws_list  = list()
    smiles_to_nswcs_map                = dict()
    for smiles in unique_gen_smiles_list:
        if pad_token in smiles:
            invalid_padded_gen_smiles_list.append(smiles)

        # Try to generate an RDKit molcular object from the SMILES string,
        # if this is None, the SMILES string cannot be made into an RDKit
        # molecular object and we assume it is invalid.
        # If this is the case, continue to next smiles
        mol = rdkit.Chem.MolFromSmiles(smiles) ## from simile to mol
        if mol is None:
            continue

        # Get the non-stereochemical washed canonical SMILES (nswcs) of the SMILES
        # If this throws a Kekulize Exception, continue to next molecule
        try:
            nswcs = cheminf.get_washed_canonical_smiles(smiles, remove_stereochemistry=True)
        except KekulizeException:
            print(f"Cannot kekulize the smiles string:\n{smiles}\n Treating it as invalid.")
            continue

        # Add the smiles and nswcs to the corresponding map/dict
        smiles_to_nswcs_map[smiles] = nswcs
        
        # If we got here, the SMILES is valid
        unique_valid_gen_smiles_list.append(smiles)

        # If the SMILES has not been in the training set, it is a novel valid smiles
        if smiles not in train_nswcs_list:
            unique_novel_valid_gen_smiles_list.append(smiles)

        # If the nswcs has been in the training set, it is a novel valid nswcs
        # Remark: The train SMILES are actually all nswcs by construction
        if nswcs not in train_nswcs_list:
            unique_novel_valid_gen_ncsws_list.append(nswcs)

    # Determine total numbers
    num_gen_smiles                    = len(gen_smiles_list)
    num_invalid_padded_gen_smiles     = len(invalid_padded_gen_smiles_list)
    num_unique_gen_smiles             = len(unique_gen_smiles_list)
    num_unique_valid_gen_smiles       = len(list(set(unique_valid_gen_smiles_list)))
    num_unique_novel_valid_gen_smiles = len(list(set(unique_novel_valid_gen_smiles_list)))
    num_unique_novel_valid_gen_nswcs  = len(list(set(unique_novel_valid_gen_ncsws_list)))

    # Display results for user
    if logger is None:
        print(f"Number of generated smiles:                    {num_gen_smiles}")
        print(f"Number of invalid padded generated smiles:     {num_invalid_padded_gen_smiles}")
        print(f"Number of unique generated smiles:             {num_unique_gen_smiles}")
        print(f"Number of unique valid generated smiles:       {num_unique_valid_gen_smiles}")
        print(f"Number of unique novel valid generated smiles: {num_unique_novel_valid_gen_smiles}")
        print(f"Number of unique novel valid generated nswcs:  {num_unique_novel_valid_gen_nswcs}")
        print()
        print(f"Invalid padding fraction (total):            {num_invalid_padded_gen_smiles/num_gen_smiles*100:.2f}%")
        print(f"Uniqueness fraction:                         {num_unique_gen_smiles/num_gen_smiles*100:.2f}%")
        print(f"Validity fraction (total):                   {num_unique_valid_gen_smiles/num_gen_smiles*100:.2f}%")
        if 0<num_unique_gen_smiles:
            print(f"Validity fraction (unique):                  {num_unique_valid_gen_smiles/num_unique_gen_smiles*100:.2f}%")
        else:
            print(f"Validity fraction (unique):                  Division-by-zero")
        if 0<num_unique_valid_gen_smiles:
            print(f"SMILES novelty fraction (unique,valid):      {num_unique_novel_valid_gen_smiles/num_unique_valid_gen_smiles*100:.2f}%")
            print(f"NSWCS novelty fraction (unique,valid):       {num_unique_novel_valid_gen_nswcs/num_unique_valid_gen_smiles*100:.2f}%")
        else:
            print(f"SMILES novelty fraction (unique,valid):      Division-by-zero")
            print(f"NSWCS novelty fraction (unique,valid):       Division-by-zero")
        if 0<num_unique_novel_valid_gen_smiles:
            print(f"NSWCS/SMILES fraction (unique,valid,novel):  {num_unique_novel_valid_gen_nswcs/num_unique_novel_valid_gen_smiles*100:.2f}%")
        else:
            print(f"NSWCS/SMILES fraction (unique,valid,novel):  Division-by-zero")
    else:
        logger.info(f"Number of generated smiles:                    {num_gen_smiles}")
        logger.info(f"Number of invalid padded generated smiles:     {num_invalid_padded_gen_smiles}")
        logger.info(f"Number of unique generated smiles:             {num_unique_gen_smiles}")
        logger.info(f"Number of unique valid generated smiles:       {num_unique_valid_gen_smiles}")
        logger.info(f"Number of unique novel valid generated smiles: {num_unique_novel_valid_gen_smiles}")
        logger.info(f"Number of unique novel valid generated nswcs:  {num_unique_novel_valid_gen_nswcs}")
        logger.info('')
        logger.info(f"Invalid padding fraction (total):            {num_invalid_padded_gen_smiles/num_gen_smiles*100:.2f}%")
        logger.info(f"Uniqueness fraction:                         {num_unique_gen_smiles/num_gen_smiles*100:.2f}%")
        logger.info(f"Validity fraction (total):                   {num_unique_valid_gen_smiles/num_gen_smiles*100:.2f}%")
        if 0<num_unique_gen_smiles:
            logger.info(f"Validity fraction (unique):                  {num_unique_valid_gen_smiles/num_unique_gen_smiles*100:.2f}%")
        else:
            logger.info(f"Validity fraction (unique):                  Division-by-zero")
        if 0<num_unique_valid_gen_smiles:
            logger.info(f"SMILES novelty fraction (unique,valid):      {num_unique_novel_valid_gen_smiles/num_unique_valid_gen_smiles*100:.2f}%")
            logger.info(f"NSWCS novelty fraction (unique,valid):       {num_unique_novel_valid_gen_nswcs/num_unique_valid_gen_smiles*100:.2f}%")
        else:
            logger.info(f"SMILES novelty fraction (unique,valid):      Division-by-zero")
            logger.info(f"NSWCS novelty fraction (unique,valid):       Division-by-zero")
        if 0<num_unique_novel_valid_gen_smiles:
            logger.info(f"NSWCS/SMILES fraction (unique,valid,novel):  {num_unique_novel_valid_gen_nswcs/num_unique_novel_valid_gen_smiles*100:.2f}%")
        else:
            logger.info(f"NSWCS/SMILES fraction (unique,valid,novel):  Division-by-zero")


    # Initialize dictionary that will 
    analysis_dict = {
        'gen_smiles_list': gen_smiles_list,
        'unique_gen_smiles_list': unique_gen_smiles_list,
        'unique_valid_gen_smiles_list': list(set(unique_valid_gen_smiles_list)),
        'unique_novel_valid_gen_smiles_list': list(set(unique_novel_valid_gen_smiles_list)),
        'unique_novel_valid_gen_nswcs_list': list(set(unique_novel_valid_gen_ncsws_list)),
        'smiles_to_nswcs_map': smiles_to_nswcs_map,
        'validity_fraction': num_unique_valid_gen_smiles/num_gen_smiles,
    }

    return analysis_dict


def get_property_distr(df:pd.DataFrame, 
                       which:str, 
                       property_name:str, 
                       num_valid_samples:int, 
                       enfore_uniqueness:bool=True) -> np.ndarray:
    """
    Return an array of property values to be extracted from a
    DataFrame containing generated molecules and their properties.
    
    Args:
        df (pandas.DataFrame): Dataframe with generated molecules
            and their properties as entries.
        which (str): Should the ground truth or predicted property
            values be extracted?
            Either the 'ground_truth' or 'predicted'.
        property_name (str): Name of the property the distribution
            should be obtained of.
        num_valid_samples (int): Number of valid samples (i.e. molecules)
            to extract from df to obtain the property distribution.
        enfore_uniqueness (bool): Ensure that all samples correspond to 
            different molecules.
            (Default: True)

    Return:
        (numpy.array) Array with the property values as entries.
    
    """
    # Construct datasets containing only the unique valid molecules
    if enfore_uniqueness:
        filtered_df = filter_unique_valid(df, num_valid_samples)
    else:
        filtered_df = df

    # Get the distribution of the property
    column  = f"{which}_{property_name}"
    return np.array(filtered_df[column])

def filter_unique_valid(df:pd.DataFrame, 
                        num_valid_samples:int) -> pd.DataFrame:
    """
    Filter a DataFrame containing generated molecules and their properties
    so that only a certain number of unique valid molecules remains.
    
    Args:
        df (pandas.DataFrame): Dataframe with generated molecules
            and their properties as entries.
        num_valid_samples (int): Number of valid samples (i.e. molecules)
            to extract from df to obtain the property distribution.

    Return:
        (pandas.DataFrame) Dataframe containing the filtered (unique) molecules
            and their properties.
    """
    # Filter by valid smiles and drop-duplicates
    filtered_df = df[df['valid']==True] 
    filtered_df = filtered_df.drop_duplicates()

    # Get the target property value
    target_property_name_value = None
    for column in df.columns:
        if column.startswith('target_'):
            target_property_name  = column
            target_property_value = list(set(df[column]))[0]
            target_property_name_value = f"{target_property_name}={target_property_value}"

    # Display stats:
    # Remark: The uniqueness fraction of the corresponding molecules by 
    # dividing the number of unique nswcs by the number of unique smiles:
    num_unique_smiles = len(list(set(filtered_df['smiles'])))
    num_unique_nswcs  = len(list(set(filtered_df['nswcs'])))
    print(target_property_name_value)
    print(f"Number of smiles (after filter): {len(filtered_df)}")
    print(f"#unique_valid-smiles:            {num_unique_smiles}")
    print(f"#unique-valid_molecules:         {num_unique_nswcs}")
    print(f"#uniqueness-fraction:            {num_unique_nswcs/num_unique_smiles:.3f}")
    print('-'*100)

    # only return the requested first number of samples (i.e. valid smiles)
    return filtered_df[:num_valid_samples]

def create_folder_if_inexistent(folder_path) -> None:
    """
    Create a folder (and its parents) if it does not exist yet. 
    
    Args:
        folder_path (str or Path): Path to the to be created (if it doesn't exist) folder.
    
    """
    if not os.path.exists(folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        print(f"Created the following inexistent folder (and any inexistent parents): {folder_path}")
