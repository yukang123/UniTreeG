# Import public modules
import rdkit
import numpy as np
from IPython.display import SVG
from numbers import Number
from rdkit import DataStructs
from rdkit.Chem import Draw, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Draw import rdMolDraw2D 
from typing import List, Optional

# Import module obtained from 'https://github.com/chembl/ChEMBL_Structure_Pipeline'
import chembl_structure_pipeline
import tdc
# from rdkit.Chem import QED
# from rdkit.Contrib.SA_Score import sascorer
from applications.molecules.MOOD_scorer.docking import get_dockingvina
import torch
import time

# Set logging level for RDKit
lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

def draw_molecule_grid(smiles_list:List[str], 
                       property_name:Optional[str]=None, 
                       property_label:str='X', 
                       **kwargs) -> object:
    """
    Draw molecules on a grid.

    Args:
        smiles_list (list of str): List of SMILES strings of
            the molecules that should be drawn on a grid.
        property_name (None or str): Name of the property to be shown next to each molecule. 
            If None, no property is displayed
            (Default: None)
        property_label (str): Label for the property to be displayed.
            (Default: 'X')
        **kwargs: Key-word arguments forwarded to Draw.MolsToGridImage().

    Return:
        (float or int): Requested property value of the molecule
            corresponding to the input SMILES string.
    
    """
    mol_list = list()
    legend = list()
    for smiles in smiles_list:
        # Generate an RDKit molecule object
        mol = rdkit.Chem.MolFromSmiles(smiles)
        mol_list.append(mol)

        # Get the wishes molecular property
        if property_name is not None:
            mol_property = get_property_value(smiles, property_name=property_name)
            if mol_property==int(mol_property):
                mol_property =int(mol_property)
                mol_annotation = str(mol_property)
            else:
                mol_annotation = f"{mol_property:.2f}"
            
            mol_annotation = r'(' + property_label + r'=' + mol_annotation + r')'

            legend.append(mol_annotation)

    img = Draw.MolsToGridImage(mol_list, legends=legend, **kwargs)
    return img

def get_property_value(smiles:str, property_name:str, strict=True, oracle=None, ignore_invalid=False, **docking_kwargs) -> Number:
    """
    Return they property value of the molecule corresponding 
    to the input SMILES string.

    Args:
        smiles (str): SMILES string of the molecule for which
            the requested property should be returned for.
        property_name (str): Property name.

    Return:
        (float or int): Requested property value of the molecule
            corresponding to the input SMILES string.
    
    """
    if property_name=='logp':
        return get_logp(smiles, addHs=True, strict=strict, ignore_invalid=ignore_invalid)
    elif property_name=='num_rings':
        return get_num_rings(smiles, strict=strict, ignore_invalid=ignore_invalid)
    elif property_name=='num_heavy_atoms':
        return get_num_heavy_atoms(smiles, strict=strict)
    elif property_name=='mol_weight':
        return get_molecular_weight(smiles)
    elif property_name=='num_tokens':
        return len(smiles), 1 ## query call does not matter
    elif property_name == "vina_docking":
        return get_docking_score(smiles, strict=strict, ignore_invalid=ignore_invalid, verify=False, **docking_kwargs)
    else:
        default_score = 0.0 if not ignore_invalid else  -np.inf #np.nan

        if oracle is None:
            try:
                # Get the requested property value
                oracle = tdc.Oracle(name=property_name)
            except:
                err_msg = f"Invalid Property Name: {property_name}."
                raise ValueError(err_msg) 
        prev_num_called = oracle.num_called
        try:
            mol_property = oracle(smiles)
            if property_name == "sa": ## [TODO] check if this is correct
                # assert get_molecular_sa(smiles) == mol_property, f"Error: {get_molecular_sa(smiles)} != {mol_property}"
                # if mol_property != 0:
                #     aa = 1
                ## revert "sa" so that higher values are better
                mol_property = (10.0 - mol_property)/9.0 if mol_property !=100 and mol_property != 0.0 else 0.0 ## the original value is between 1 and 10
            if mol_property == 0.0:  
                mol_property = default_score ## revise to default_score
            new_num_called = oracle.num_called
            return mol_property, new_num_called - prev_num_called
        except Exception as e:
            print("[WARNING] Error while getting the property value for SMILES: ", e)
            return default_score, 0 # 0.0

    # elif property_name=="qed":
    #     return get_molecular_qed(smiles)
    # else:
    #     err_msg = f"The passed property_name '{property_name}' does not correspond to an expected property name."
    #     raise ValueError(err_msg)
#### TODO: need further debugging #######
def get_property_value_all(
        smiles_list:list, 
        property_name:str, #vina_target="parp1",
        strict=True, oracle=None, ignore_invalid=False, **docking_kwargs
    ) -> Number:
    """
    Return a list of property values of the molecules corresponding 
    to the input SMILES string list.

    Args:
        smiles_list (list): SMILES string list of the molecules for which
            the requested property should be returned for.
        property_name (str): Property name.

    Return:
        list(float or int): a list of Requested property values of the molecules
            corresponding to the input SMILES strings.
    
    """
    num_query_call_all = 0
    metric_value_list = []
    if property_name=='logp':
        for smiles in smiles_list:
            metric_value, num_query_call = get_logp(smiles, addHs=True, strict=strict, ignore_invalid=ignore_invalid) #for smiles in smiles_list
            num_query_call_all += num_query_call
            metric_value_list.append(metric_value)
        return metric_value_list, num_query_call_all

    elif property_name=='num_rings':
        for smiles in smiles_list:
            metric_value, num_query_call = get_num_rings(smiles, strict=strict, ignore_invalid=ignore_invalid)
            num_query_call_all += num_query_call
            metric_value_list.append(metric_value)
        return metric_value_list, num_query_call_all
    
    elif property_name=='num_heavy_atoms':
        for smiles in smiles_list:
            metric_value, num_query_call = get_num_heavy_atoms(smiles, strict=strict)
            num_query_call_all += num_query_call
            metric_value_list.append(metric_value)
        return metric_value_list, num_query_call_all
    
    elif property_name=='mol_weight':
        for smiles in smiles_list:
            metric_value, num_query_call = get_molecular_weight(smiles)
            num_query_call_all += num_query_call
            metric_value_list.append(metric_value)
        return metric_value_list, num_query_call_all
    
    elif property_name=='num_tokens':
        for smiles in smiles_list:
            num_tokens = len(smiles)
            metric_value_list.append(num_tokens)
            num_query_call_all += 1
        return metric_value_list, num_query_call_all
    
    elif property_name == "vina_docking":
        return get_docking_score_all(smiles_list, ignore_invalid=ignore_invalid, **docking_kwargs) #vina_target=vina_target, 
    
    else:
        default_score = 0.0 if not ignore_invalid else -np.inf

        if oracle is None:
            try:
                # Get the requested property value
                oracle = tdc.Oracle(name=property_name)
            except:
                err_msg = f"Error while getting the property '{property_name}' for the SMILES strings."
                raise ValueError(err_msg) 
        prev_num_called = oracle.num_called

        mol_property = np.array(oracle(smiles_list))
        if property_name == "sa": ## [TODO] check if this is correct
            ## revert and normalize "sa" so that higher values are better
            label = np.bitwise_or((mol_property == 100), (mol_property == 0))
            mol_property = (10.0 - mol_property)/9.0 * (1-label) + 0.0 * label 
        
        mol_property = np.where(mol_property == 0.0, default_score, mol_property)  # replace 0.0 with default_score
        new_num_called = oracle.num_called
        return mol_property.tolist(), new_num_called - prev_num_called

def draw_molecule(smiles:str) -> None:
    """
    Draw the molecule corresponding to the passed smiles string.
    
    Args:
        smiles (str): SMILES string of the molecule to be drawn.

    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
        raise ValueError(err_msg)

    # Generate a drawer
    drawer = rdMolDraw2D.MolDraw2DSVG(300,300)
    drawer.drawOptions().addStereoAnnotation = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Display the drawn molecule
    display( SVG(drawer.GetDrawingText()) )

def get_washed_canonical_smiles(smiles:str, 
                                remove_stereochemistry:bool=False) -> str:
    """
    'Wash' the input SMILES string and return its canonical version.
    
    Args:
        smiles (str): (Canonical) SMILES string.
        remove_stereochemistry (bool): Should we also remove stereochemistry?
            (Default: False)

    Return:
        (str): Washed (canonical) SMILES string.
    
    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol = rdkit.Chem.MolFromSmiles(smiles)

    # Standardize (neutralize) the molecular object
    st_mol = chembl_structure_pipeline.standardize_mol(mol)

    # Remove salt and solvent
    st_mol, _ = chembl_structure_pipeline.get_parent_mol(st_mol)

    # If multiple fragments remain, take the one with the most heavy atoms
    st_mol_frags = rdkit.Chem.GetMolFrags(st_mol, asMols=True, sanitizeFrags=False)
    if 1 < len(st_mol_frags):
        st_mol_frags = sorted(
            st_mol_frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True
        )
        st_mol = st_mol_frags[0]
        
    # If we should remove the stereochemistry from the molecule, remove it
    if remove_stereochemistry:
        rdkit.Chem.RemoveStereochemistry(st_mol) 

    # Get the canonical SMILES string of the 'washed' molecular object and return it
    smiles = rdkit.Chem.MolToSmiles(st_mol, canonical=True)
    return rdkit.Chem.CanonSmiles(smiles)


def get_molecular_weight(smiles:str) -> float:
    """
    Return the molecular weight of the molecule generated by the input SMILES string.
    
    Args:
        smiles (str): SMILES string for which the molecular weight 
            should be returned for.

    Return:
        (float): Molecular weight of the molecule corresponding 
            to the input SMILES string.

    """
    # Suppress constant printouts while standardizing
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    # Generate an RDKit molecule object
    mol = rdkit.Chem.MolFromSmiles(smiles)  ### TODO: how to handle the case of mol=None?

    # Determine and return the molecular weight of the RDKit molecule object
    return rdkit.Chem.rdMolDescriptors.CalcExactMolWt(mol), 1 ## query call does not matter

def get_num_rings(smiles:str, strict=True, ignore_invalid=False) -> int:
    """
    Return the number of rings of a molecule corresponding to the input SMILES string.
    
    Args:
        smiles (str): SMILES string for which the number of
            rings should be returned for.

    Return:
        (int): Number of rings of the molecule corresponding 
            to the input SMILES string.

    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    default_score = -1 if not ignore_invalid else -np.inf
    num_query_call = 0
    if mol is None:
        if strict:
            err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
            raise ValueError(err_msg)
        else:
            return default_score, num_query_call #-1 #0 ## TODO: check the value

    return rdMolDescriptors.CalcNumRings(mol), num_query_call # number of rings

def get_logp(smiles:str, 
             addHs:bool=True, strict=True, ignore_invalid=False) -> float:
    """
    Return the lipophilicity (LogP) of a molecule corresponding to the input SMILES string.
    
    Args:
        smiles (str): SMILES string for which the lipophilicity 
            should be returned for.
        addHs (bool): Optional boolean flag to specify if H-atoms 
            should be added in the computation or not.
            (Default: True)

    Return:
        (float): Lipophilicity of the molecule corresponding 
            to the input SMILES string.

    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    default_score = -4 if not ignore_invalid else -np.inf
    num_query_call = 0
    if mol is None:
        if strict:
            err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
            raise ValueError(err_msg)
        else:
            return default_score, num_query_call #-4  ## TODO: check the value
    
    num_query_call +=1
    return MolLogP(mol, includeHs=addHs), num_query_call # logp calculated by Crippen's approach

def get_num_heavy_atoms(smiles:str, strict=True) -> int:
    """
    Return the number of heavy atoms of a molecule corresponding to the input SMILES string. 
    
    Args:
        smiles (str): SMILES string for which the number of
            heavy atoms should be returned for.

    Return:
        (int): Number of heavy-atoms of the molecule corresponding 
            to the input SMILES string.

    """
    mol = rdkit.Chem.MolFromSmiles(smiles)
    num_query_call = 0
    if mol is None:
        if strict:
            err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
            raise ValueError(err_msg)
        else:
            return 0, num_query_call

    num_query_call += 1
    return mol.GetNumHeavyAtoms(), num_query_call # Number of heavy atoms

def get_morgan_fingerprint(smiles:str, 
                           fp_radius:int=2, 
                           fp_size:int=1024) -> object:
    """
    Return the Morgan fingerprint of molecule corresponding to the input SMILES string. 
    
    Args:
        smiles (str): SMILES string for which the Morgan
            fingerprint should be returned for.
        fp_radius (int): Morgan finger print (mfp) radius.
            (Default: 2)
        fp_size (int): Morgan finger print (mfp) size (i.e. length).
            (Default: 1024)

    Return:
        (object): Morgan fingerprint as RDKit fingerprint object.
    """
    # Get the molecule from the smiles string
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
        raise ValueError(err_msg)

    # Initialize a fingerprint generator and generate a fingerprint from the molecule object
    # that is returned
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=fp_radius, fpSize=fp_size)
    return fp_generator.GetFingerprint(mol)

def get_tanimoto_similarities(smiles_list:List[str], 
                              which:str='average', 
                              mfp_radius:int=2, 
                              mfp_size:int=1024) -> np.ndarray:
    """
    (1) Determine the pairwise Tanimoto similarities (using Morgan fingerprints) 
    of the molecules corresponding to the passed list of SMILES string. 
    (2) Per molecule return either the average (which='average') or maximal 
    (which='maximal') Tanimoto similarity to all other molecules.

    Args:
        smiles_list (list): List of SMILES strings.
        which (str): Which Tanimoto similarity to return per molecule;
            - 'average': Return average Tanimoto similarity of one molecule
                to all the other molecules.
            - 'maximal': Return maximal Tanimoto similarity of one molecule
                to all the other molecules (i.e. return the similarity to most
                similar other molecule).
            (Default: 'average')
        mfp_radius (int): Morgan finger print (mfp) radius.
            (Default: 2)
        mfp_size (int): Morgan finger print (mfp) size (i.e. length).
            (Default: 1024)
    
    Return:
        (numpy.array): 1D numpy array of shape (#SMILES,) holding 'average' or 
            'maximal' Tanimoto similarity of each SMILES to all the other SMILES 
            strings.
    
    """
    ### Step (0)
    # Determine fingerprints for each of the SMILES strings
    fp_list = [get_morgan_fingerprint(smiles, fp_radius=mfp_radius, fp_size=mfp_size) for smiles in smiles_list]
    
    ### Step (1)
    # Determine the pairwise Tanimoto similarities (pwts) between all molecules
    # Inspired by:
    # https://stackoverflow.com/questions/51681659/how-to-use-rdkit-to-calculte-molecular-fingerprint-and-similarity-of-a-list-of-s
    # Remark: Exclude the last molecule ('len(fp_list)-1)') because all pairwise
    #         similarities have already been determined for it in the iterations
    #         of the previous molecules.
    pwts_matrix = np.zeros((len(smiles_list), len(smiles_list)))
    for mol_index in range(len(fp_list)-1):
        # Determine similarities between the current molecule and
        # all the 'molecules in the list after this molecule'
        mol_pwts_list = DataStructs.BulkTanimotoSimilarity(fp_list[mol_index], fp_list[mol_index+1:]) # returns as 'list' object

        # Assign the pairwise Tanimoto similarities to their values in
        # the pairwise Tanimoto similairities matrix
        pwts_matrix[mol_index, mol_index+1:] = np.array(mol_pwts_list)

    # The pairwise Tanimoto similairities matrix is upper diagonal matrix (with zero on diagonal), 
    # transform it to a symmetric matrix with np.nan on the diagonal.
    pwts_matrix = pwts_matrix + pwts_matrix.T
    pwts_matrix = pwts_matrix + np.diag(np.ones(pwts_matrix.shape[0])*np.nan)

    ### Step (2)
    if which=='average':
        return np.nanmean(pwts_matrix, axis=0)
    elif which=='maximal':
        return np.nanmax(pwts_matrix, axis=0)
    else:
        err_msg = f"Input 'which' must be either 'average' or 'maximal', got '{which}' instead."
        raise ValueError(err_msg)
    

from rdkit.Chem import inchi

vina_target_list = ['parp1', 'fa7', '5ht1b', 'jak2', 'braf']
def get_docking_score(
        smiles:str, 
        vina_target:str='parp1',
        strict:bool=True, 
        ignore_invalid:bool=False,
        # verbose:bool=False,
        verify:bool=False,
        use_gpu:bool=True,
        num_cpu:int=1,
        gpu_thread:int=8000,
        gpu_parallel:bool=True,
        eval_batch_size:int=16,
        base_temp_dir:str='/tmp/yukang',
        **kwargs
        ) -> float:
    
    mol = rdkit.Chem.MolFromSmiles(smiles)
    num_query_call = 0
    default_score = 0 if not ignore_invalid else -np.inf
    if mol is None:
        if strict:
            err_msg = f"The SMILES string '{smiles}' cannot be made into an RDKit molecule object."
            raise ValueError(err_msg)
        else:
            return default_score, num_query_call

    assert vina_target in vina_target_list, f"Target {vina_target} is not in the target list: {vina_target_list}"
    # vina_score = get_docking_scores(vina_target, [mol], verbose=True)
    # return vina_score[0]

    if verify:
        verify_smiles = rdkit.Chem.MolToSmiles(mol)
        if verify_smiles != smiles:
            print(f"[WARNING] SMILES verification failed: {verify_smiles} != {smiles}")
        inchi1 = inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(smiles))
        inchi2 = inchi.MolToInchiKey(rdkit.Chem.MolFromSmiles(verify_smiles))
        assert (inchi1 == inchi2)

    dockingvina = get_dockingvina(
        vina_target, 
        num_cpu=num_cpu, use_gpu=use_gpu, gpu_thread=gpu_thread, gpu_parallel=gpu_parallel, eval_batch_size=eval_batch_size,
        base_temp_dir=base_temp_dir
    )

    # smiles = [standardize_smiles(mol) for mol in mols]
    # smiles_valid = [smi for smi in smiles if smi is not None]
    smiles_valid = [smiles]
    scores = - np.array(dockingvina.predict(smiles_valid))
    if verify:
        dockingvina = get_dockingvina(vina_target, num_cpu=num_cpu, use_gpu=use_gpu, gpu_thread=gpu_thread, gpu_parallel=gpu_parallel, eval_batch_size=eval_batch_size) 
        verify_scores = - np.array(dockingvina.predict([verify_smiles]))
        if not np.isclose(scores[0], verify_scores[0]):
            print(f"[WARNING] Docking score verification failed: {scores[0]} != {verify_scores[0]}")
    scores = list(np.clip(scores, 0, None))
    num_query_call += 1
    return scores[0], num_query_call


def get_docking_score_all(
        smiles_list:list[str], 
        vina_target:str='parp1',
        ignore_invalid:bool=False,
        use_gpu:bool=True,
        num_cpu:int=1,
        gpu_thread:int=8000,
        gpu_parallel:bool=True,
        eval_batch_size:int=16,
        base_temp_dir:str='/tmp/yukang',
        verbose:bool=False,
        count_extreme_score: bool=False,
        **kwargs
        # verify:bool=False
        ) -> float:
    
    assert vina_target in vina_target_list, f"Target {vina_target} is not in the target list: {vina_target_list}"

    default_score = 0 if not ignore_invalid else -np.inf
    scores = np.zeros(len(smiles_list)) 
    indices_valid = []
    smiles_valid = []
    for i, smi in enumerate(smiles_list):
        mol = rdkit.Chem.MolFromSmiles(smi)
        if mol is None:
            scores[i] = default_score
        else:
            smiles_valid.append(smi)
            indices_valid.append(i)
    valid_docking_cal_tag = np.array([False] * len(smiles_list), dtype=bool)
    if len(smiles_valid) > 0:
        if verbose:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        dockingvina = get_dockingvina(vina_target, num_cpu=num_cpu, use_gpu=use_gpu, gpu_thread=gpu_thread, gpu_parallel=gpu_parallel, eval_batch_size=eval_batch_size, base_temp_dir=base_temp_dir)
        valid_scores = - np.array(dockingvina.predict(smiles_valid)) ### default: -99.9 for docking failure
        if count_extreme_score:
            valid_docking_cal_tag[np.array(indices_valid)[valid_scores != -99.9]] = True
        else:
            valid_docking_cal_tag[indices_valid] = True
        valid_scores = list(np.clip(valid_scores, 0, None))
        scores[indices_valid] = valid_scores
        if verbose:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            print(f"Docking scores computed in {end_time - start_time:.2f} seconds for {len(smiles_valid)} valid SMILES ({(end_time - start_time) / len(smiles_valid)}).")
    num_query_call = len(indices_valid) # number of valid smiles
    if count_extreme_score:
        return scores, num_query_call, valid_docking_cal_tag #num_failure_docking, num_zero_scores
    return scores, num_query_call



def get_docking_score_repeat_mean(
        smiles, repeat=48, #64,
        ignore_invalid:bool=False,
        vina_target:str='parp1',
        use_gpu:bool=True,
        num_cpu:int=1,
        gpu_thread:int=8000,
        gpu_parallel:bool=True,
        eval_batch_size:int=16,
        base_temp_dir:str='/tmp/yukang',
        exclude_failure_score:bool=False,
        # exclude_zero_score: bool=False,
        **kwargs
    ):
    smiles_list = [smiles] * repeat
    scores, num_valid_smiles, valid_docking_cal_tag = get_docking_score_all(
            smiles_list, vina_target=vina_target, ignore_invalid=ignore_invalid,
            use_gpu=use_gpu, num_cpu=num_cpu, gpu_thread=gpu_thread,
            gpu_parallel=gpu_parallel, eval_batch_size=eval_batch_size,
            base_temp_dir=base_temp_dir, count_extreme_score=True,
        )
    if ignore_invalid and num_valid_smiles == 0:
        raise ValueError(f"[Error] {smiles} is an invalid SMILES.")

    print(f"Original Docking score calculation: {repeat - np.sum(valid_docking_cal_tag)} / {repeat} failed dockings.")
    if exclude_failure_score:
        scores = scores[valid_docking_cal_tag]
        smiles_list = [smiles] * len(scores)
        if len(scores) < repeat:
            print(f"[WARNING] Excluding failure docking calculations, {len(scores)} scores remain out of {repeat} repeats.")
    assert len(scores) > 0, f"All docking score calculations are invalid for SMILES: {smiles}"
    mean_score = np.mean(scores) #if len(scores) > 0 else 0.0
    print(f"Mean docking score for {smiles} over {len(scores)} repeats: {mean_score:.2f}")
    return mean_score, len(smiles_list)  # return the mean score and the number of queries