# Import public modules
import ast
import ml_collections
import os
import yaml
from pathlib import Path
from typing import List, Optional

def update_without_overwrite(dict1:dict, 
                             dict2:dict) -> dict:
    """
    Inplace update operation that will add key-value pairs of dict2 to dict1 if
    a key does not exist in dict1 (i.e. without overwriting any existing keys).

    Args:
        dict1 (dict): First dictionary that will be updated by second dictionary.
        dict2 (dict): Second dictionary used to update first dictionary (without overwriting).

    Return:
        (dict): Updated first dictionary.
    
    """
    # Remark: dict.update() is an inplace operation.
    dict1.update({key: value for key, value in dict2.items() if key not in dict1})
    return dict1

def load_cfg_from_yaml_file(file_path:str) -> ml_collections.ConfigDict:
    """
    Load an ml_collections.ConfigDict object holding
    configurations from a .yaml file and return it.

    Args:
        file_path (str): Path to the .yaml configurations file.

    Return:
        (ml_collections.ConfigDict): Loaded config dictionary.
    
    """
    if os.path.isfile(file_path)==False:
        err_msg = f"No yaml file found in: {file_path}"
        raise FileNotFoundError(err_msg)

    with open(file_path) as stream:
        try:
            file_content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Transform the config dict to an ml collections ConfigDict
    cfg = ml_collections.ConfigDict(file_content)
    
    return cfg

def save_dict_to_yaml_file(file_path:str, 
                           cfg:ml_collections.ConfigDict) -> None:
    """
    Save the content of an ml_collections.ConfigDict object
    to a .yaml file.

    Args:
        file_path (str): Path to the to be saved .yaml file.
        cfg (ml_collections.ConfigDict): To be saved config dictionary.
    
    """
    with open(file_path, 'w') as outfile:
        yaml.dump(dict(cfg), outfile, default_flow_style=False)

def parse_overrides(overrides:dict) -> dict:
    """
    Parse overrides that is a string of override-configs separated by '|' 
    in the form '<name1>=<value1>|<name2>=<value2>|...'
    and return them as dictionary.

    Args:
        overrides (dict): Overrides dictionary to be parsed.

    Return:
        (dict): Parsed overrides dictionary.

    """
    # If overrides is an empty string, return an empty dictionary
    if overrides=='':
        return {}
    
    # Strip '"'
    overrides = overrides.strip('"')

    # Loop over all overrides and construct the dictionary containing the overrides
    overrides_list = overrides.split('|')
    overrides_dict = dict()
    for override in overrides_list:
        override_split = override.split('=')
        
        # Extract override name and value
        name  = override_split[0]
        if len(override_split)==1:
            err_msg = f"An override must be in the form '<name>=<value>' but got an override '{override}'."
            raise ValueError(err_msg)
        elif len(override_split)==2:
            value = override_split[1]
        else:
            # There are "=" in the configuration value, so recreate
            # the value by combining the "on = splitted" pieces
            value = '='.join(override_split[1:])

        # Check that the name is not already a key of the overrides dictionary
        if name in overrides_dict:
            err_msg = f"Got two overrides for '{name}', which is not allowed."
            raise ValueError(err_msg)

        # In case the value is a stringified list/dict, cast it to a list/dict
        is_stringified_list = value.startswith('[') and value.endswith(']')
        is_stringified_dict = value.startswith('{') and value.endswith('}')
        if is_stringified_list or is_stringified_dict:
            value = ast.literal_eval(value)
            if isinstance(value, list):
                # Cast items to numbers
                _value = list()
                for item in value:
                    try:
                        item = float(item)
                        if int(item)==item:
                            item = int(item)
                    except ValueError:
                        pass
                    _value.append(item)
                value = _value
            if isinstance(value, dict):
                # Cast dict-values to numbers
                _value = dict()
                for key, val in value.items():
                    try:
                        val = float(val)
                        if int(val)==val:
                            val = int(val)
                    except ValueError:
                        pass
                    _value[key] = val
                value = _value
        elif value=='True':
            value = True
        elif value=='False':
            value = False
        elif value in ['None']:
            value = None
        else:
            # Try to cast the value to a number
            try:
                value = float(value)
                # If the int version of the value is equal to the value, cast it to an int
                if int(value)==value:
                    value = int(value)
            except ValueError:
                pass
            

        # Add the name-value pair
        overrides_dict[name] = value

    # overrides might contain keys of the form {'x.y.z': value}, cast
    # this to a deep dictionary of the form {'x':{'y':{'z': value}}}
    return deep_dict_from_dict(overrides_dict)

def deep_dict_from_dict(d:dict) -> dict:
    """
    Parse a dictionary containing keys of the form {'x.y.z': value}
    to a deep dictionary of the form {'x':{'y':{'z':value}}} using
    recursion.

    Args:
        d (dict): Input dictionary to be parsed.

    Return:
        (dict of dicts): Deep dictionary parsed 
            from input dictionary.
    """
    dd = dict()
    for key, value in d.items():
        key_split = key.split('.')
        if len(key_split)==1: # Base condition
            main_key = key_split[0]
            dd[main_key] = value
        else: # Recursion
            main_key = key_split[0]
            sub_key = '.'.join(key_split[1:])
            sub_d   = deep_dict_from_dict({sub_key: value})
            if main_key in dd:
                dd[main_key].update(sub_d)
            else:
                dd[main_key] = sub_d

    return dd

def update_dirs_in_cfg(cfg:ml_collections.ConfigDict, 
                       outputs_dir:str, 
                       dont_update:list=[]) -> None:
    """
    Update the directories (that are entries of cfg) in the config dictionary 
    'cfg' by replacing the passed outputs directory 'outputs_dir'.

    Args:
        cfg (ml_collections.ConfigDict): Config dict
            whose directories should be updated. 
            Remark: cfg is changed in place.
        outputs_dir (str): Output directory to be updated in
            the directories that are entries of the config file.
        dont_update (list of str): Optional list of directory names
            (that are entries of cfg) that should not be updated.
            (Default: [] <=> i.e., update all directories)
    
    """
    cfg['outputs_dir'] = outputs_dir
    if 'checkpoints_dir' not in dont_update:
        cfg['checkpoints_dir'] = str(Path(outputs_dir, 'checkpoints'))
    if 'configs_dir' not in dont_update: 
        cfg['configs_dir'] = str(Path(outputs_dir, 'configs'))
    if 'figs_save_dir' not in dont_update:
        cfg['figs_save_dir'] = str(Path(outputs_dir, 'figs_saved'))
    if 'models_save_dir' not in dont_update:
        cfg['models_save_dir'] = str(Path(outputs_dir, 'models_saved'))
    if 'models_load_dir' not in dont_update:
        cfg['models_load_dir'] = str(Path(outputs_dir, 'models_saved'))

def find_different_configs(dict1:ml_collections.ConfigDict, 
                           dict2:ml_collections.ConfigDict) -> None:
    """
    Find all entries that differ in two (ml_collections.ConfigDict) 
    config dictionaries and print them.

    Args:
        dict1 (ml_collections.ConfigDict): First (config) dictionary.
        dict2 (ml_collections.ConfigDict): Second (config) dictionary.

    """
    # Check that all keys are the same and throw an error otherwise
    if set(dict1.keys()).symmetric_difference(set(dict2.keys())):
        raise KeyError("Keys do not match.")
    
    # Loop over all keys in the first dictionary
    for key in dict1:
        # Check that the type of the key-values are the same
        if type(dict1[key])!=type(dict2[key]):
            raise TypeError("The type of the values is different.")
        
        # If the value of the first key is of type ml_collections.ConfigDict,
        # recursively call the function on both values
        if isinstance(dict1[key], ml_collections.config_dict.config_dict.ConfigDict):
            find_different_configs(dict1[key], dict2[key])
        else:
            # Otherwise, check that the values match and if not
            # display what their values are.
            if dict1[key]!=dict2[key]:
                print(f"{key}: {dict1[key]}")
                print(f"{key}: {dict2[key]}")
                print('-'*100)