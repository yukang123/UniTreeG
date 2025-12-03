# Import public modules
import collections
import ml_collections
import scipy
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pathlib import Path
from sklearn import model_selection
from typing import Optional, Tuple

# Import custom modules
from . import plotting
from . import utils

class MoleculesDataHandler(object):
    def __init__(self, 
                 cfg:ml_collections.ConfigDict, 
                 dataset_df:pd.DataFrame, 
                 make_figs:bool=True, 
                 save_figs:bool=False, 
                 figs_save_dir:Optional[str]=None, 
                 logger:Optional[object]=None,
                 debug:bool=False) -> None:
        """
        Args:
            cfg (ml_collections): Config dictionary.
            dataset_df (pandas.DataFrame): Dataset as pandas.DataFrame.
            make_figs (bool): Should figures be made during preprocessing?
                (Default: True)
            save_figs (bool): Should figures be saved during preprocessing?
                (Default: False)
            figs_save_dir (None or str): Path to folder in which figures should
                be saved in (in case save_figs=True).
                If None, no figures will be saved (even if save_figs=True).
                (Default: None)
            logger (None or object): Logger object.
                If None no logger is used.
                (Default: None)
        
        """

        # Assign configs to class attributes
        self.validation_train_ratio      = cfg.validation_train_ratio
        self.filter_order                = cfg.filter_order
        self.filter_range_dict           = cfg.filter_range_dict
        self.property_data_sampling_dict = cfg.property_data_sampling_dict
        self.torch_data_property_names   = cfg.torch_data_property_names
        self.random_seed_split           = cfg.random_seed_split
        self.make_figs                   = make_figs
        self.save_figs                   = save_figs
        self.figs_save_dir               = figs_save_dir
        self.logger                      = logger
        
        # Filter dataset_df
        self.dataset_df = self.filter_dataset_df(dataset_df)

        # Determine the maximal number of tokens (over all nswcs)
        self.max_num_tokens = max([len(nswcs) for nswcs in self.dataset_df['nswcs']])
        self.display(f"Maximum number of tokens (over all nswcs): {self.max_num_tokens}")

        # Determine the unique tokens in the nswcs list
        unique_tokens_set = set()
        for nswcs in self.dataset_df['nswcs']:
            unique_tokens_set.update(set([token for token in nswcs]))

        self.display(f"Unique tokens (#{len(unique_tokens_set)}): {unique_tokens_set}")

        # Initialize a smiles encoder (that can also be used for decoding)
        self.smiles_encoder = SMILESEncoder(unique_tokens_set, 
                                            max_num_tokens=self.max_num_tokens)

        if debug:
            return
        # Split data into train-validation (i.e. no test set here)
        ix_train, ix_validation = ix_train_test_split(len(self.dataset_df), 
                                                      self.validation_train_ratio, 
                                                      random_state=self.random_seed_split, 
                                                      stratify=None)
        self.ix_dict = {'train': ix_train, 'validation': ix_validation}

        # Generate a dataframe for each set
        self.subset_df_dict = {set_name: self.dataset_df.iloc[set_ix] for set_name, set_ix in self.ix_dict.items()}

        # Loop over the properties and their sampling specifications
        # Remark: This will add property-train subsets to self.subset_df_dict
        for property_name, sampling_specs in self.property_data_sampling_dict.items():
            self.sample_property_train_subset(property_name, **sampling_specs)

        # Display the number of datapoints in all subsets
        for set_name, set_df in self.subset_df_dict.items():
            self.display(f"#{set_name}: {len(set_df)}")
        
        # Generate Dict-Datasets for each subset with the encoded nswcs as 'x' and the other data attributes by their name
        self.torch_dataset_dict = dict()
        for set_name, subset_df in self.subset_df_dict.items():
            # Encode the nswcs as matrix for the current set
            encoded_nswcs_matrix = np.vstack([self.smiles_encoder(nswcs) for nswcs in subset_df['nswcs']])
            
            # Other data attributes
            set_attributes_dict = {property_name: torch.tensor(list(subset_df[property_name])) for property_name in self.torch_data_property_names}

            # Construct the dict-dataset
            torch_dataset = DictDataset(x=torch.tensor(encoded_nswcs_matrix, dtype=torch.int), 
                                        **set_attributes_dict)
            self.torch_dataset_dict[set_name] = torch_dataset

        # Fit the distribution of the properties specified in 'self.torch_data_property_names'
        # of training data using a normal distribution and save the sigmas
        self.train_property_sigma_dict = dict()
        for property_name in self.torch_data_property_names:
            res_dict = self.fit_normal_to_property_distribution(property_name, set_name='train')
            self.train_property_sigma_dict[property_name] = float(res_dict['sigma'])
    
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

    def to(self, device:object) -> None:
        """
        Map the datasets to the specified device.

        Args:
            device (object): Device the data should be mapped to.
        """
        for set_name in self.torch_dataset_dict:
            self.torch_dataset_dict[set_name].to(device)

    @property
    def token_alphabet_size(self) -> int:
        """ Return the token alphabet size (as integer). """
        return self.smiles_encoder.token_alphabet_size
    
    @property
    def pad_token(self) -> str:
        """ Return the pad token (as string). """
        return self.smiles_encoder.pad_token
    
    @property
    def pad_token_index(self) -> int:
        """ Return the (token alphabet) index of the pad token (as integer). """
        return self.smiles_encoder.pad_token_index

    @property
    def device(self) -> object:
        """ Return the device the datasets are located on. """
        return self.torch_dataset_dict['train'].device
    
    @property
    def max_num_rings_train(self) -> int:
        """ Return the maximal number of rings of the molecules in the training set (as integer). """
        if 'num_rings' not in self.subset_df_dict['train'].columns:
            err_msg = f"Cannot return maximal number of rings of molecules in training set, because there is no 'num_rings' information in the dataset."
            raise ValueError(err_msg)
        
        return max(self.subset_df_dict['train']['num_rings'])
    
    @property
    def train_num_tokens_freq_dict(self) -> dict:
        """
        Construct and return the number-of-tokens frequency dictionary of the train set.

        Return:
            Dictionary with 'number of tokens' as dict-keys and associated frequencies
            (i.e. number of molecules with a certain "number of tokens") within the train set as 
            dict-values in the form {<num_tokens>: <frequency>, ...}.

        """
        # Generate a list containing the number of tokens per nswcs in the train set
        num_tokens_per_train_nswcs_list = [len(nswcs) for nswcs in self.subset_df_dict['train']['nswcs']]
        
        # Count the number of occurances of a certain "number of tokens". 
        # Remark: collections.Counter will return a dict-like counter object with "number of tokens" 
        #         as keys and #occurances (i.e. #molecules with the "number of tokens") as values.
        num_tokens_to_counts_map = collections.Counter(num_tokens_per_train_nswcs_list)

        # Divide the counts by the total number of counts (over all "number of tokens")
        # to obtain the frequencies of the "number of tokens".
        total_counts = np.sum(list(num_tokens_to_counts_map.values()))
        if total_counts==0:
            err_msg = f"Cannot calculate frequencies as the total number of nswcs (i.e. molecule) counts was zero for the train set."
            raise ValueError(err_msg)

        # Determine the frequencies and return the resulting dictionary
        # Remark: Sort the "number of tokens" in increasing order first
        num_tokens_list = list(num_tokens_to_counts_map.keys())
        num_tokens_list.sort()
        return {str(num_tokens): float(num_tokens_to_counts_map[num_tokens]/total_counts) for num_tokens in num_tokens_list}

    def filter_dataset_df(self, dataset_df:pd.DataFrame) -> pd.DataFrame:
        """
        Filter the input dataset and return the result.

        Args:
            dataset_df (pandas.DataFrame): Dataset to be filtered.
        
        Return:
            (pandas.DataFrame): Filtered dataset.

        """
        # Filter the dataset, only keeping molecules with values in certain range
        for property_name in self.filter_order:
            # Check that the property name is a key of self.filter_range_dict and throw an error if it isn't
            if property_name not in self.filter_range_dict:
                err_msg = f"Cannot filter property '{property_name}' because it is not a key of 'self.filter_range_dict'."
                raise KeyError(err_msg)

            filter_range = self.filter_range_dict[property_name]
            
            # Check that property name corresponds to column of dataset_df
            if property_name not in dataset_df.columns:
                err_msg = f"Cannot filter on '{property_name}' because no column with this name is in dataset_df."
                raise ValueError(err_msg)
            
            # Check that the filter range contains a first element that is smaller than second
            if filter_range[1]<filter_range[0]:
                err_msg = f"The filter range for '{property_name}' contains second entry that is smaller than first entry, got filter range {filter_range}"
                raise ValueError(err_msg)

            if self.make_figs:
                fig = plt.figure()
                ax = plt.gca()
                # Plot pre-filtering
                _hist = plotting.custom_hist(dataset_df[property_name], bins=100, color='teal', histtype='stepfilled', label='pre-filter', ax=ax)

            # Filter
            dataset_df = dataset_df[(filter_range[0]<=dataset_df[property_name]) & (dataset_df[property_name]<=filter_range[1])]

            if self.make_figs:
                # Plot post-filtering
                plotting.custom_hist(dataset_df[property_name], bins=_hist[1], color='blue', histtype='stepfilled', label='post-filter', ax=ax)

                # Plot filter range lines
                plt.vlines(filter_range[0], 0, max(_hist[0]), color='grey', ls='--', label='lower-threshold')
                plt.vlines(filter_range[1], 0, max(_hist[0]), color='grey', ls=':', label='upper-threshold')

                # Set plot specs
                plt.xlabel(f'{property_name}')
                plt.ylabel('#Molecules')
                plt.legend()
                plt.show()

                # Save the figure
                if self.save_figs and self.figs_save_dir is not None:
                    file_path = str(Path(self.figs_save_dir, f"filtered_distribution_{property_name}.png"))
                    fig.savefig(file_path)

        return dataset_df
    
    def sample_property_train_subset(self, 
                                     property_name:str, 
                                     fraction:float, 
                                     stratify:bool=True, 
                                     use_max_num_bins:bool=False, 
                                     seed:int=120) -> None:
        """
        Sample (without replacement) a subset of the training data (fraction) for the property-predictor.

        To counter-act issues when training a property-predictor, sample subsets (without replacement) of the train set to
        obtain a subset for which the property labels are approximately stratified (i.e. the property distribution is 
        approx. uniform of the train data range).

        Args:
            property_name (str): Name of the property to 
                sample a train subset from. 
            fraction (float): Fraction in [0, 1] to sample from
                the train set to obtain the property subset. 
            stratify (bool): Should sampling be stratified based
                on the property?
                (Default: True)
            use_max_num_bins (bool): In case the property values
                are binned (e.g., naturally for categorical properties
                or by design for continuous quantities) for stratification, 
                should the maximal number of bins be used?
                (Default: False) 
            seed (int): Seed for random state used in subsampling.
                (Default: 120)

        """
        # Set the random seed for the sampling
        utils.set_random_seed(seed)
        
        # Copy the train set
        train_set_df = self.subset_df_dict['train'].copy(deep=True)

        # Determine the number of datapoints in this subset
        num_samples = int(len(train_set_df)*fraction)
        
        if stratify:
            # Apply stratified sampling
            property_train_subset_df = self.sample_stratified_df(property_name, train_set_df, num_samples=num_samples, use_max_num_bins=use_max_num_bins)
        else:
            # Uniformly sample (without replacement) the requested number of samples from the processed DataFrame
            property_train_subset_df = train_set_df.sample(n=num_samples, replace=False)

        # Plot the sampled property train subset
        if self.make_figs:
            fig = plt.figure()
            ax = plt.gca()
            plt.title(f"Property distribution of subsampled training set (fraction: {fraction*100}%)")
            plotting.custom_hist(property_train_subset_df[property_name], ax=ax, ec='black')
            plt.xlabel(property_name)
            plt.show()
            if self.save_figs and self.figs_save_dir is not None:
                file_path = str(Path(self.figs_save_dir, f"train_subset_subsampled_{property_name}.png"))
                fig.savefig(file_path)

        # Assign this to the subset dictionary
        self.subset_df_dict[f"train_{property_name}"] = property_train_subset_df

        
    def sample_stratified_df(self, 
                             column:str, 
                             df:pd.DataFrame, 
                             num_samples:int, 
                             use_max_num_bins:bool=False) -> pd.DataFrame:
        """
        Bin the data in a certain column and stratifiy w.r.t. these bins. 
        
        Args:
            column (str): Column name within input dataset
                based on whose values rows should be sampled
                from in a stratified manner.
            df (pandas.DataFrame): Input dataset to sample from.
            num_samples (int): Number of samples (i.e., rows)
                to sample from the input dataset.
            use_max_num_bins (bool): In case the column values
                are binned (e.g., naturally for categorical column values
                or by design for continuous quantities) for stratification, 
                should the maximal number of bins be used?
                (Default: False)  

        Return:
            (pandas.DataFrame): Stratified-sampled dataset.
        
        """
        # Bin the data, either using the maximum number of bins (e.g. for 'num_rings' and other property)
        max_num_bins = len(set(df[column]))
        if use_max_num_bins:
            # Use max-number of bins
            num_bins = max_num_bins

            # Determine the number of samples per bin
            num_samples_per_bin = int(np.ceil(num_samples/num_bins))

            # Bin the DataFrame for the column values
            df, bin_edges, bin_center_column = self.bin_df_on_column_values(df, column, num_bins)
        else:
            # Use bisection-variant to find smallest number of bins that allow 
            num_bins1 = min(max_num_bins, 100)
            num_bins2 = num_bins1
            
            while True:
                num_bins = num_bins2
                
                if num_bins<=1:
                    err_msg = f"Using one bin does not lead to necessary number of samples, decrease fraction (current value)."
                    raise ValueError(err_msg)

                # Determine the number of samples per bin
                num_samples_per_bin = int(np.ceil(num_samples/num_bins))

                # Bin the DataFrame for the column values
                df, bin_edges, bin_center_column = self.bin_df_on_column_values(df, column, num_bins)
                
                # Determine the minimal points in each bin
                min_num_pnts_per_bin = min(collections.Counter(df[bin_center_column]).values())

                #if num_samples_per_bin<min_num_pnts_per_bin:
                if min_num_pnts_per_bin<num_samples_per_bin:
                    num_bins1 = num_bins2
                    num_bins2 = int(np.floor(num_bins2/2))
                else:
                    num_bins2 = num_bins1 - int(np.floor((num_bins1-num_bins2)/2))
                
                # Break if the two number of bins values are the same
                if num_bins1==num_bins2:
                    break

        # Use stratified sampling (without replacement) to draw the same number of samples per bin (i.e. strata)
        stratified_df = df.groupby(bin_center_column, observed=True).sample(n=num_samples_per_bin, replace=False)

        # Uniformly sample (without replacement) the requested number of samples from the processed DataFrame
        # Remark: Note that stratified_df has already approximatively (but maybe a little bit more that) num_samples entries.
        #         This step is only necessary to get the exact requested number of samples and will not strongly affect
        #         stratified distribution.
        sub_sampled_stratified_df = stratified_df.sample(n=num_samples, replace=False)

        return sub_sampled_stratified_df
    
    def bin_df_on_column_values(self, 
                                df:pd.DataFrame, 
                                column:str, 
                                num_bins:int) -> Tuple[pd.DataFrame, np.ndarray, str]:
        """
        Bin the dataset on column values.

        Args:
            df (pandas.DataFrame): Input dataset to be binned.
            column (str): Name of the column whose values should
                be binned.
            num_bins (int): Number of bins to use.

        Returns:
            (pandas.DataFrame): Binned dataset.
            (np.ndarray): Edges of the binned column values.
            (str): Name of the newly created column containing
                the centers of the binned column values.
                This name will be constructed from the input
                column as '<column>_bin_center'.
        
        """
        # Determine the min and max values of the property in the train set
        # Remark: Subtract and add tiny values to avoid edge-effects
        min_vals = np.min(df[column])-1e-10
        max_vals = np.max(df[column])+1e-10

        # Determine the bin_edges and bin_centers
        bin_edges   = np.linspace(min_vals, max_vals, num_bins+1)
        bin_centers = bin_edges[:-1]+np.diff(bin_edges)

        # Assign datapoints to the bins
        bin_center_column = f"{column}_bin_center"
        df[bin_center_column] = pd.cut(df[column], bin_edges, labels=bin_centers)
        
        # Ensure that all points were assigned to bins
        # Remark: If they could not be assign their value in column 'bin_center_column' will be None/null
        if 0<len(df[df[bin_center_column].isnull()]):
            err_msg = f"Some points could not be assigned to a bin for the column '{column}'."
            raise ValueError(err_msg)

        return df, bin_edges, bin_center_column
    
    def fit_normal_to_property_distribution(self, 
                                            property_name:str, 
                                            set_name:str) -> dict:
        """
        Fit a normal distribution to the distribution of the specified property
        of the specified set.

        Args:
            property_name (str): Name of the property whose values should 
                be fitted.
            set_name (str): Name of the set for which the property values
                should be fitted for.

        Return:
            (dict): Results of the normal fit as dictionary of the form:
                {
                    'mu': <fitted-normal-mean>, 
                    'sigma': <fitted-normal-standard-deviation
                }
        
        """
        # Check that the set_name is allowed
        if set_name not in self.subset_df_dict:
            err_msg = f"The 'set_name' must be one of the following: {list(self.subset_df_dict.keys())}"

        # Check that the property_name is a column of self.subset_df_dict[set_name]
        if property_name not in self.subset_df_dict[set_name].columns:
            err_msg = f"The property '{property_name}' is not a column name of self.subset_df_dict['{set_name}']. The column names are: {self.subset_df_dict['train'].columns}"
            raise ValueError(err_msg)

        # Fit a normal distribution to the train distribution of a property
        property_vals = self.subset_df_dict[set_name][property_name]
        cost_p = lambda p: -np.mean(scipy.stats.norm(loc=p[0], scale=p[1]).logpdf(property_vals))
        p_init = [2.5, 0.5]
        soln  = scipy.optimize.minimize(cost_p, p_init)
        p_opt = soln.x

        # Plot the fit
        if self.make_figs:
            fig = plt.figure()
            # Plot data distribution
            plotting.custom_hist(property_vals, bins=100, density=True, color='b', label='Data')

            # Plot fitted distribution
            t = np.linspace(min(property_vals), max(property_vals), 1000)
            opt_pdf_t = scipy.stats.norm(loc=p_opt[0], scale=p_opt[1]).pdf(t)
            plt.plot(t, opt_pdf_t, 'r-', label='Normal-fit')
            plt.xlabel(property_name)
            plt.ylabel('Density')
            plt.legend()
            plt.show()

            # Save the figure
            if self.save_figs and self.figs_save_dir is not None:
                file_path = str(Path(self.figs_save_dir, f"fitted_{set_name}_distribution_{property_name}.png"))
                fig.savefig(file_path)

        return {'mu': p_opt[0], 'sigma': p_opt[1]}

def ix_train_test_split(num_datapoint:int, 
                        test_size:float, 
                        random_state:Optional[int]=None, 
                        stratify:Optional[ArrayLike]=None) -> Tuple[ArrayLike, ArrayLike]:
    """
    Return train/test splitted datapoint indices.

    Args:
        num_datapoint (int): Number of datapoints in the unsplit dataset.
        test_size (float): Fraction of test set as value in [0, 1].
        random_state (None or int): Random state (i.e., seed) to be 
            used for the split.
            If None, do not control randomness.
            (Default: None)
        stratify (None or ArrayLike): Array-like object with values
            to be stratified on.
            If None, do not stratify.
            (Default: None)

    Remark: For more details concerning the inputs 'test_size', 'random_state',
            and 'stratify', checkout the official documentation of
            sklearn.model_selection.train_test_split:
            https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.train_test_split.html

    Return:
        (ArrayLike): Array of train datapoint indices.
        (ArrayLike): Array of test datapoint indices.
    """
    # Create an index array for all datapoints
    ix_all = np.arange(0, num_datapoint)

    # Use train-test splitting on this index array
    ix_train, ix_test, _, _ = model_selection.train_test_split(ix_all, 
                                                               ix_all, 
                                                               test_size=test_size, 
                                                               random_state=random_state, 
                                                               stratify=stratify)
    
    return ix_train, ix_test

class SMILESEncoder(object):
    def __init__(self, 
                 unique_tokens_set:set, 
                 max_num_tokens:int, 
                 pad_token:str='$') -> None:
        """
        Args:
            unique_tokens_set (set): Set of all unique tokens
                that could appear in the to be encoded SMILES 
                strings. 
            max_num_tokens (int): Maximum token length that
                could appear in the be encoded SMILES strings.
            pad_token (str): Pad token symbol.
                (Default: '$')
        """
        # Assign inputs to class attributes
        self.unique_tokens_list = list(unique_tokens_set)
        self.unique_tokens_list.sort()
        self.max_num_tokens     = max_num_tokens
        self.pad_token          = pad_token

        # Check that the pad token is not in the list of unique tokens
        if self.pad_token in self.unique_tokens_list:
            err_msg = f"Cannot use '{self.pad_token}' as pad token as this token already appears in the unpadded smiles."
            raise ValueError(err_msg)
        
        # Add the pad token at the end of the unique token list
        self.unique_tokens_list.append(self.pad_token)

        # Generate a mapping from token to index (i.e. categorical state label) and back
        self.index_to_token_map = {index: token for index, token in enumerate(self.unique_tokens_list)}
        self.token_to_index_map = {token: index for index, token in self.index_to_token_map.items()}

    @property
    def token_alphabet_size(self) -> int:
        """ Return the token alphabet size (as integer). """
        return len(self.index_to_token_map)
    
    @property
    def pad_token_index(self) -> int:
        """ Return the (token alphabet) index of the pad token (as integer). """
        return self.token_to_index_map[self.pad_token]

    def encode(self, smiles:str) -> np.ndarray:
        """
        Numerically encode SMILES string (string -> np.ndarray),
        while padding it to the maximum number of tokens.
        
        Args:
            smiles (str): SMILES string to be (numerically) encoded.

        Return:
            (np.ndarray): Numerically encoded SMILES string.
        
        """
        # Pad the SMILES string if necessary
        if len(smiles)==self.max_num_tokens:
            padded_smiles = smiles
        elif len(smiles)<self.max_num_tokens:
            padding = ''.join([self.pad_token]*(self.max_num_tokens-len(smiles)))
            padded_smiles = smiles + padding
        else:
            err_msg = f"Cannot encode SMILES strings longer than max_num_tokens={self.max_num_tokens}, but got the following SMILES with {len(smiles)} tokens: {smiles}"
            raise ValueError(err_msg)
        
        # Map padded smiles to numpy array
        return np.array([self.token_to_index_map[token] for token in padded_smiles])

    def __call__(self, smiles:str) -> np.ndarray:
        """
        Wrapper around method 'encode'.
        
        Args:
            smiles (str): SMILES string to be (numerically) encoded.

        Return:
            (np.ndarray): Numerically encoded SMILES string.
        
        """
        return self.encode(smiles)

    def decode(self, encoded_smiles:np.ndarray):
        """
        Decode numerically encoded SMILES string (np.ndarray->string),
        while removing any padding.
        
        Args:
            encoded_smiles (np.ndarray): Numerically encoded SMILES string
                that should be decoded.

        Return:
            (str): Decoded SMILES string.
        
        """
        # Decode the encoded smiles that potentially has padding at the end
        try:
            decoded_smiles = ''.join([self.index_to_token_map[index] for index in encoded_smiles])
            # Remove any padding token from the right side (i.e. end) of the decoded smiles
            return decoded_smiles.rstrip(self.pad_token)
        except KeyError:
            return 'None'

class DictDataset(torch.utils.data.Dataset):
    """
    Define a custom dataset that returns a dictionary with dictionary-keys 
    'x' if sliced where the dictionary-values will correspond to the 
    sliced x data values.
    The same can be done for additional key-value pairs within kwargs.
    (E.g., one could pass 'y=<torch.tensor>' that would add also a 'y' entry).
    """
    def __init__(self, 
                 x:torch.tensor, 
                 device:Optional[object]=None, 
                 **kwargs) -> None:
        """
        Args:
            x (torch.tensor): 2D torch tensor of shape (#datapoints, #x-features).
            device (None or object): Device the data should be mapped to.
                If None no device is specified.
                (Default: None)
            **kwargs: Additional entries as key-value pairs.
        
        """
        if device is None:
            # Get the device
            device = torch.device('cpu')

        # Assign x and y to the corresponding class attributes
        self.device    = device
        self.x         = x.to(self.device)
        self.vars_dict = {key: value.to(self.device) for key, value in kwargs.items()}

    def to(self, device:object) -> None:
        """
        Map the data to the specified device.
        
        Args:
            device (object): The device the data
                should be mapped to.

        """
        # Update the device class attribute
        self.device = device

        # Map everything to the wished device
        self.x         = self.x.to(self.device)
        self.vars_dict = {key: value.to(self.device) for key, value in self.vars_dict.items()}

    def __len__(self) -> int:
        """ Return the number of datapoints (as integer). """
        # Remark: self.x should have shape (#datapoints, #x-features)
        return self.x.shape[0]

    def __getitem__(self, ix:int) -> dict:
        """
        Implement slicing. 
        
        Args:
            ix (int): Datapoint index.

        Return:
            (dict): Item correspondig to 
                datapoint index.
            
        """
        # Cast ix to a list if it is a tensor
        if torch.is_tensor(ix):
            ix = ix.tolist()        

        # Return a dictionary containing the data slices for ix
        ix_data_dict = {'x': self.x[ix]}
        ix_data_dict.update({var_name: var_values[ix] for var_name, var_values in self.vars_dict.items()})
        return ix_data_dict