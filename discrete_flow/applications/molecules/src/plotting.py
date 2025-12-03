# Import public modules
import collections
import copy
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.typing import ArrayLike
from typing import Iterable, Optional, List, Tuple

# Import custom modules
from . import cheminf
from . import utils
import tdc
from tqdm import tqdm
def plot_gen_vs_train_distribution(property_name:str, 
                                   train_subset_df:pd.DataFrame,
                                   gen_smiles_list:List[str],
                                   figsize:Tuple[float, float]=(7.0, 7.0),
                                   ax:Optional[object]=None,
                                   xlabel=None, **docking_eval_kwargs) -> None:
    """
    Plot the property distribution of generated SMILES strings
    v.s. the SMILES strings in the train (sub)set.

    Args:
        property_name (str): Property name. 
        train_subset_df (pandas.DataFrame): Train (sub)set.
        gen_smiles_list (list of str): List of generated SMILES strings.
        figsize (2-tuple of floats): Figure size.
            (Defualt: (7.0, 7.0))
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)

    """
    # # Ensure that the property name is a column name in the rain subset
    # if property_name not in train_subset_df.columns:
    #     err_msg = f"The property '{property_name}' is not a column of 'train_subset_df', which has the columns: {train_subset_df.columns}."
    #     raise ValueError(err_msg)

    # If ax was not passed, make a figure and extract the axis from it
    make_fig = False
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        make_fig = True

    if property_name in train_subset_df.columns:
        # Plot the train set property distribution
        custom_hist(train_subset_df[property_name], color='blue', label='Train', density=True, ax=ax)

    # Determine the property distribution of the generated SMILES strings and plot it
    gen_property_values = []
    for smiles in tqdm(gen_smiles_list):
        if property_name == "vina_docking":
            ground_truth_property_value, _ = cheminf.get_docking_score_repeat_mean(smiles, ignore_invalid=False, **docking_eval_kwargs) #
        else:
            ground_truth_property_value, _ = cheminf.get_property_value(smiles, property_name=property_name, strict=True, ignore_invalid=False, **docking_eval_kwargs)
        gen_property_values.append(ground_truth_property_value)
    # gen_property_values = [cheminf.get_property_value(smiles, property_name, strict=True, ignore_invalid=False, **docking_eval_kwargs)[0] for smiles in gen_smiles_list]
    custom_hist(gen_property_values, color='red', alpha=0.75, label='Generated', density=True, ax=ax)

    # Set plot specs
    if xlabel is None:
        ax.set_xlabel(property_name)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()

    if make_fig:
        plt.show()
    return gen_property_values

def plot_train_distribution(property_name:str, 
                            train_subset_df:pd.DataFrame,
                            figsize:Tuple[float, float]=(7.0, 7.0),
                            ax:Optional[object]=None,
                            xlabel=None) -> None:
    """
    Plot the property distribution of generated SMILES strings
    v.s. the SMILES strings in the train (sub)set.

    Args:
        property_name (str): Property name. 
        train_subset_df (pandas.DataFrame): Train (sub)set.
        gen_smiles_list (list of str): List of generated SMILES strings.
        figsize (2-tuple of floats): Figure size.
            (Defualt: (7.0, 7.0))
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)

    """
    # # Ensure that the property name is a column name in the rain subset
    # if property_name not in train_subset_df.columns:
    #     err_msg = f"The property '{property_name}' is not a column of 'train_subset_df', which has the columns: {train_subset_df.columns}."
    #     raise ValueError(err_msg)

    # If ax was not passed, make a figure and extract the axis from it
    make_fig = False
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        make_fig = True

    if property_name in train_subset_df.columns:
        # Plot the train set property distribution
        train_property_values = train_subset_df[property_name]
    else:
        try:
            oracle = tdc.Oracle(name=property_name)
        except ValueError:
            oracle = None
        train_property_values = []
        for smiles in tqdm(train_subset_df["nswcs"]):
            property_values = cheminf.get_property_value(smiles, property_name, oracle=oracle)
            train_property_values.append(property_values)
        # train_property_values = cheminf.get_property_value_all(list(train_subset_df["nswcs"]), property_name, oracle=oracle)
        
    custom_hist(train_property_values, color='blue', label='Train', density=True, ax=ax)

    # Set plot specs
    if xlabel is None:
        ax.set_xlabel(property_name)
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()

    if make_fig:
        plt.show()
    return train_property_values

def make_correlation_plot(predictor_model_name:str,
                          orchestrator:object,
                          set_name:str='validation', 
                          t_eval:float=1.0, 
                          seed:int=42,
                          discrete_y_data:bool=False,
                          num_bins_x_axis:int=100,
                          num_bins_y_axis:int=100,
                          normalize:bool=False) -> object:
    """
    Make a correlation plot for ground truth vs. predicted property values.

    Args:
        predictor_model_name (str): Name of the property-predictor model
            that indirectly also specifies the property.
        orchestrator (object): Orchestrator object as defined in 'factory.py'.
        set_name (str): Name of the set the SMILES are taken from.
            (Default: 'validation')
        t_eval (float): At which time to evaluate the prediction.
            (Default: 1.0)
        seed (int): Seed for random states in case property-prediction is
            non-deterministic.
            (Default: 42)
        discrete_y_data (bool): Should the data-properties (y-data) be discretized
            (e.g., in case it is continuous but the predicted properties are discrete)
            (Default: False)
        num_bins_x_axis (int): Number of bins on x-axis (i.e., ground truth properties axis).
            (Default: 100)
        num_bins_y_axis (int): Number of bins on y-axis (i.e., predicted properties axis).
            (Default: 100)
        normalize (bool): Normalize the Z-values of the 2D histogram (that is used in case
            that both the ground truth and predicted property values are discrete/discretized).
            (Default: False)

    Return:
        (object): Matplotlib figure object.

    """
    # Check that t_eval is in [0, 1]
    if t_eval<0 or 1<t_eval:
        err_msg = f"The input time must be in [0, 1], got value {t_eval} instead."

    # Get the manager
    manager = orchestrator.manager

    # Ensure that the predictor model is valid
    if predictor_model_name not in manager.models_dict:
        err_msg = f"The predictor model '{predictor_model_name}' name is not a model of the manager, which has models with names: {list(manager.models_dict.keys())}"
        raise KeyError(err_msg)

    # Get the predictor model and set it into eval mode
    predictor_model = manager.models_dict[predictor_model_name]
    predictor_model.eval()

    # Set the seed
    utils.set_random_seed(seed)

    # Get the dataloader of the subset
    if set_name in orchestrator.dataloader_dict:
        dataloader = orchestrator.dataloader_dict[set_name]
    else:
        err_msg = f"Set name '{set_name}' is unexpected. Use one of the following: {list(orchestrator.dataloader_dict.keys())}"
        raise ValueError(err_msg)

    # Loop over the batch data
    y_data_list = list()
    y_pred_list = list()
    for batch_data in dataloader:
        # Get x1
        x1 = batch_data['x']

        # Use the same one fixed time for evaluation on all batch points
        t = manager.get_t(t_eval, x1) # (B,)

        # Sample xt
        xt = manager.sample_xt(x1, t) # (B, D)

        # Differ cases depending on the output type of the predictor model 
        # (i.e. is the model output a probability or property value)
        # Generate a copy of batch_data and set 'x' to 'xt'
        batch_data_t      = copy.deepcopy(batch_data)
        batch_data_t['x'] = xt
        if predictor_model.output_type=='class_distribution':
            probs = manager.models_dict[predictor_model_name](batch_data_t, t)
            y_pred = torch.argmax(probs, dim=-1)
        else:    
            y_pred = manager.models_dict[predictor_model_name](batch_data_t, t)        

        y_pred_list += list(utils.to_numpy(y_pred).squeeze())
        
        # Get the data and append it
        y_data       = batch_data[predictor_model.y_guide_name]
        y_data_list += list(utils.to_numpy(y_data).squeeze())

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(f"{predictor_model_name}@t={t_eval} ({set_name} set)")
    
    ### Correlation plot (subplot 1)
    ax = axs[0]
    if discrete_y_data:
        y_data_values = np.array(y_data_list)
        y_pred_values = np.array(y_pred_list)

        # Find all unique y-data values and sort them
        unique_y_data_values = np.unique(y_data_values)
        unique_y_data_values = np.sort(unique_y_data_values)

        # The number of bins is given by the number of unique y-data values
        num_bins_x_axis = len(unique_y_data_values)
        
        # Determine the bins (for prediction, but data has naturally these bins because it is discrete)
        diffs = np.diff(unique_y_data_values)
        bin_edges_x_axis = list(unique_y_data_values[:-1]+diffs/2)
        bin_edges_x_axis = [unique_y_data_values[0]-diffs[0]/2] + bin_edges_x_axis
        bin_edges_x_axis = bin_edges_x_axis + [unique_y_data_values[-1]+diffs[-1]/2]

        if num_bins_y_axis is None:
            num_bins_y_axis = num_bins_x_axis
            num_bins_y_axis = bin_edges_x_axis
        else:
            # Bin the y-predicted values (over all y-data values)
            _hist_pred = np.histogram(y_pred_values, num_bins_y_axis)
            bin_edges_y_axis = _hist_pred[1]
        
        extent = [bin_edges_x_axis[0], bin_edges_x_axis[-1], bin_edges_y_axis[0], bin_edges_y_axis[-1]]

        # Assign values to the matrix representing bins
        Z = np.zeros((num_bins_x_axis, num_bins_y_axis))
        for x_axis_index, unique_y_data_value in enumerate(unique_y_data_values):
            ix = np.where(y_data_values==unique_y_data_value)
            local_y_pred_values = y_pred_values[ix]
            _local_hist_pred = np.histogram(local_y_pred_values, bins=bin_edges_y_axis)
            counts = np.array(_local_hist_pred[0])
            if normalize:
                counts = counts/float(np.sum(counts).squeeze())
            Z[x_axis_index, :] = counts

        ax.imshow(Z.T, aspect='auto', origin='lower', extent=extent)

    else:
        ax.plot(y_data_list, y_pred_list, color='hotpink', marker='o', alpha=0.5, ls='', ms=0.5)
        
        xy_lim = [min(y_data_list+y_pred_list), max(y_data_list+y_pred_list)]
        
        ax.plot([xy_lim[0], xy_lim[1]], [xy_lim[0], xy_lim[1]], color='k', ls='-', lw=0.5)
        ax.set_xlabel(r'$y_{data}$')
        ax.set_ylabel(r'$y_{pred}$')
        ax.set_xlim(xy_lim)
        ax.set_ylim(xy_lim)


    ### Error plot (subplot 2)
    ax = axs[1]
    # Determine the absolute differences in y_data and y_pred and plot them
    y_abs_diff_list = list( np.abs(np.array(y_data_list)-np.array(y_pred_list)) )
    _hist = ax.hist(y_abs_diff_list, bins=100, color='hotpink', alpha=0.5)
    
    # Determine the mean of the absolute differences (and its standard error) and plot it as vertical line
    mean_y_abs_diff = np.mean(y_abs_diff_list)
    err_y_abs_diff  = np.std(y_abs_diff_list)/np.sqrt(len(y_abs_diff_list))
    ax.vlines(mean_y_abs_diff, 0, max(_hist[0]), label=r'<|$y_{data}$-$y_{pred}$|>'+f"={mean_y_abs_diff:.4f}"+r'$\pm$'+f"{err_y_abs_diff:.4f}")

    # Set plot psecs
    ax.legend()
    ax.set_xlabel(r'|$y_{data}$-$y_{pred}$|')
    ax.set_xlim([0, max(y_abs_diff_list)])                 

    plt.show()

    return fig

def custom_hist(x: Iterable, 
                ax:Optional[object]=None, 
                set_x_ticks:bool=False, 
                **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    """
    Make custom histogram plot where if x are all integer values a bar plot is shown.
    Otherwise a normal histogram is shown.

    Args:
        x (Iterable): Iterable of values to plot a histogram of.
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)
        set_x_ticks (bool): Set x-ticks in function 'make_custom_bar_plot'?
            (Default: False)
        **kwargs: Keyword-arguments forwarded to function 'make_custom_bar_plot'.

    Return:
        (2-tuple of array-like objects): 2-tuple of the form (<hist-counts>, <hist-edges>).

    """
    # If ax was not passed, make a figure and extract the axis from it
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Check that x is an iterable
    if not isinstance(x, Iterable):
        err_msg = f"The input must be an iterable, got type {type(x)} instead."
        raise ValueError(err_msg)
    
    # Throw an error if x is empty
    if len(x)==0:
        err_msg = f"The input cannot be an empty iterable."
        raise ValueError(err_msg)

    # First check if x only contains integers or not
    x_contains_only_ints = True
    for item in x:
        # If an item cast to an integer is equal to its float expression, 
        # it must be an integer
        if int(item)!=float(item):
            x_contains_only_ints = False

    # Differ cases
    if x_contains_only_ints:
        # Transform all elements in x to true integers
        _x = [int(item) for item in x]

        # Make a custom bar plot an return the results
        return make_custom_bar_plot(_x, ax=ax, set_x_ticks=set_x_ticks, **kwargs)
    else:
        # Make a histogram plot and return the result
        return ax.hist(x, **kwargs)
    
def make_custom_bar_plot(x:Iterable, 
                         ax:Optional[object]=None, 
                         set_x_ticks:bool=False, 
                         **kwargs) -> object:
    """
    Make custom bar plot.

    Args:
        x (Iterable): Iterable of values to plot bars from.
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)
        set_x_ticks (bool): Set x-ticks?
            (Default: False)
        **kwargs: Keyword-arguments forwarded to function 'matplotlib.pyplot.bar'.

    Return:
        (2-tuple of array-like objects): 2-tuple of the form (<bar-counts>, <bar-edges>).

    """
    # If ax was not passed, make a figure and extract the axis from it
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # kwargs could contain some keys that are not kwargs of plt.bar.
    # Remove these keys here as we will pass kwargs below.
    for delete_key in ['bins', 'histtype']:
        kwargs.pop(delete_key, None)

    # Check if 'density' has been defined in the kwargs and
    # if so read it out (while deleting it at the same time).
    # Remark: We delete 'density' because it is not a kwargs 
    #         of plt.bar.
    density = kwargs.pop('density', False)

    # Count the number of occurences of the elements of x
    loc_to_counts_map = collections.Counter(x)

    # The locations are the unique elements of x and correspond
    # to the dictionary-keys of 'loc_to_counts_map'
    locs = list(loc_to_counts_map.keys())
    locs.sort()

    # Get the associated counts
    counts = [loc_to_counts_map[loc] for loc in locs]

    # Transform both to numpy arrays
    locs   = np.array(locs)
    counts = np.array(counts)

    # Determine the bar width
    if len(locs)==1:
        # If there is only one location, use a width of 1 as default
        width = 1
    else:
        # If there is more than one location, determine the bar width as the 
        # smallest difference in the (sorted) locs
        width = np.min(np.diff(locs))

    # If density is True, multiply the counts by the (global) width resulting in the
    # probability mass of each bar, and then normalize them to one over all bars.
    if density:
        counts = counts.astype(float)*width
        sum_counts = np.sum(counts)
        if sum_counts==0:
            err_msg = f"Cannot normalize the counts as their sum is zero."
            raise ValueError(err_msg)
        
        counts /= sum_counts

    # Make a bar plot with the locs and counts
    ax.bar(locs, counts, width=width, **kwargs)

    # Set the xticks to the locs if requested
    if set_x_ticks:
        ax.set_xticks(locs)

    # Generate a similar output to plt.hist in the form of a 2-tuple (counts, bar_edges) 
    # containing two tuples where the N+1 bar_edges are computed from the N locations as 
    # [locs[0]-width/2, locs[0]+width/2, ..., locs[N-1]+width/2]
    bar_edges  = [float(locs[0])-width/2]
    bar_edges += list(locs+width/2)

    return (tuple(counts), tuple(bar_edges))

def plot_num_rings_distr(x_l:ArrayLike, 
                         x_r:ArrayLike, 
                         bin_centers:ArrayLike,
                         color_l:str='b',
                         color_r:str='r',
                         label_l:str='1',
                         label_r:str='2',
                         ax:Optional[object]=None, 
                         panel_label:Optional[str]=None,
                         panel_label_rel_xy:Tuple[float, float]=[0.95, 0.95],
                         panel_label_ha:str='left',
                         panel_label_fontweight:Optional[float]=None,
                         x_label:Optional[str]=None,
                         show_x_ticks:bool=True,
                         show_y_ticks:bool=True,
                         fs_dict:dict={'axis': 20, 'leg': 17.5, 'ticks': 15, 'title': 22.5},
                         show_legend:bool=True,
                         leg_loc:str='best',
                         leg_handletextpad:Optional[float]=None,
                         leg_borderpad:Optional[float]=None,
                         leg_framealpha:Optional[float]=None,
                         leg_vertical_labelspacing:Optional[float]=None,
                         leg_handlelength:Optional[float]=None,
                         bar_width_scale:float=0.4,
                         **kwargs) -> None:

    """
    Make a plot to compare two different number of rings distributions.

    Args:
        x_l (ArrayLike): First (i.e., 'left') distribution.
        x_r (ArrayLike): Second (i.e., 'right') distribution.
        bin_centers (ArrayLike): Bin centers for the histograms of
            created for both distributions.
        color_l (str): Color of first (i.e., 'left') distribution.
            (Default: 'b')
        color_r (str): Color of second (i.e., 'right') distribution.
            (Default: 'r')
        label_l (str): Legend-label of first (i.e., 'left') distribution.
            (Default: '1')
        label_r (str): Legend-label of second (i.e., 'right') distribution.
            (Default: '2')
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)  
        panel_label (None or str): Label for the panel.
            If None, no panel labels is displayed
            (Default: None)
        panel_label_rel_xy (2-tuple of floats): Relative xy-position of the
            panel label.
            (Default: [0.95, 0.95])
        panel_label_ha (str): Horizontal alignment of panel label.
            (Default: 'left')
        panel_label_fontweight (None or float): Which fontweight to use for the 
            panel label.
            If None, use default font weight for panel label.
            (Default: None)
        x_label (None or str): Label for x-axis. 
            If None, do not display x-axis label.
            (Default: None)
        show_x_ticks (bool): Should the x-axis ticks be displayed?
            (Default: True)
        show_y_ticks (bool): Should the y-axis ticks be displayed?
            (Default: True)
        fs_dict (dict): Dictionary holding the different fontsizes
            with keys: ['axis', 'leg', 'ticks', 'title'].
            (Default: {'axis': 20, 'leg': 17.5, 'ticks': 15, 'title': 22.5})
        show_legend (bool): Should the legend be displayed?
            (Default: True)
        leg_loc (str): Legend location.
            (Default: 'best')
        leg_handletextpad (None or float): The pad between the legend handle 
            and text, in font-size units.
            If None, use the default padding.
            (Default: None)
        leg_borderpad (None or float): The fractional whitespace inside the 
            legend border, in font-size units.
            If None, use default fractional whitespace.
            (Default: None)
        bar_width_scale (float): Scale for the widths of the shown bars 
            relative to the actual histogram-bin widths.
            (Default: 0.4)
        **kwargs: Keyword-arguments forwarded to 'matplotlib.pyplot.hist' function.
    
    """
    if ax is None:
        ax = plt.gca()

    # Get the unique x values for both sets as bin centers
    if bin_centers is None:
        bin_centers = list(set(list(x_l)+list(x_r)))
        bin_centers.sort()
        bin_centers = np.array(bin_centers)

    # Calculate the bin edges
    diffs = np.diff(bin_centers)
    bin_edges = list(bin_centers[:-1]+diffs/2)
    bin_edges = [bin_centers[0]-diffs[0]/2] + bin_edges + [bin_centers[-1]+diffs[-1]/2]
    bin_edges = np.array(bin_edges)

    # Bin the data to obtain bars
    _hist_l      = np.histogram(x_l, bins=bin_edges)
    _hist_r      = np.histogram(x_r, bins=bin_edges)
    bar_counts_l = np.array(_hist_l[0], dtype=np.float64)
    bar_counts_r = np.array(_hist_r[0], dtype=np.float64)

    # Normalize the bar counts to obtain bar probabilities
    bar_probs_l = bar_counts_l/np.sum(bar_counts_l)
    bar_probs_r = bar_counts_r/np.sum(bar_counts_r)

    # Get the minimal difference
    min_diff  = float(np.min(diffs))
    bar_width = min_diff*bar_width_scale
    bar_shift = bar_width/2

    # Plot the bars
    for bin_index, bin_center in enumerate(bin_centers):
        # Determine the bar centers
        bar_center_l = bin_center-bar_shift
        bar_center_r = bin_center+bar_shift

        # Make labels
        if bin_index==0:
            _label_l = label_l
            _label_r = label_r
        else:
            _label_l = None
            _label_r = None

        # Plot the bars
        ax.bar(bar_center_l, bar_probs_l[bin_index], width=bar_width, color=color_l, label=_label_l, **kwargs)
        ax.bar(bar_center_r, bar_probs_r[bin_index], width=bar_width, color=color_r, label=_label_r, **kwargs)

    ###################################################
    ### Set plot specs
    ###################################################
    if x_label is None:
        ax.set_xlabel('')
    else:
        ax.set_xlabel(x_label, fontsize=fs_dict['axis'])

    if show_x_ticks:
        ax.set_xticks(bin_centers)
    else:
        ax.set_xticks([])

    ax.set_ylabel('')
    if show_y_ticks:
        ax.set_yticks([0, 1])
    else:
        ax.set_yticks([])

    ax.tick_params(axis='both', labelsize=fs_dict['ticks'])
    
    # Set x limits
    x_lims = [min(bin_centers)-0.5, max(bin_centers)+0.5]
    ax.set_xlim(x_lims)

    # Set y-limits
    y_lims = [0, 1]
    ax.set_ylim(y_lims)

    # Add a panel label if requested
    if panel_label is not None:
        panel_label_x = x_lims[0]+(x_lims[1]-x_lims[0])*panel_label_rel_xy[0]
        panel_label_y = y_lims[0]+(y_lims[1]-y_lims[0])*panel_label_rel_xy[1]
        ax.text(panel_label_x, panel_label_y, panel_label, fontsize=fs_dict['text'], ha=panel_label_ha, va='top', fontweight=panel_label_fontweight)

    # Show legend if requested
    if show_legend:
        ax.legend(fontsize=fs_dict['leg'], loc=leg_loc, handletextpad=leg_handletextpad, borderpad=leg_borderpad, labelspacing=leg_vertical_labelspacing, handlelength=leg_handlelength, framealpha=leg_framealpha)

def plot_logp_distr(x_l:ArrayLike, 
                    x_r:ArrayLike, 
                    bin_edges:ArrayLike,
                    color_l:str='b',
                    color_r:str='r',
                    label_l:str='1',
                    label_r:str='2',
                    ax:Optional[object]=None, 
                    panel_label:Optional[str]=None,
                    panel_label_rel_xy:Tuple[float, float]=[0.95, 0.95],
                    panel_label_fontweight:Optional[float]=None,
                    x_label:Optional[str]=None,
                    x_ticks:Optional[ArrayLike]=None,
                    show_y_ticks:bool=True,
                    fs_dict:dict={'axis': 20, 'leg': 17.5, 'ticks': 15, 'title': 22.5},
                    show_legend:bool=True,
                    leg_loc:str='best',
                    leg_handletextpad:Optional[float]=None,
                    leg_borderpad:Optional[float]=None,
                    leg_framealpha:Optional[float]=None,
                    leg_vertical_labelspacing:Optional[float]=None,
                    leg_handlelength:Optional[float]=None,
                    **kwargs) -> Tuple[float, object]:
    """
    Make a plot to compare two different LogP distributions.

    Args:
        x_l (ArrayLike): First (i.e., 'left') distribution.
        x_r (ArrayLike): Second (i.e., 'right') distribution.
        bin_edges (ArrayLike): Bin edges for the histograms of
            created for both distributions.
        color_l (str): Color of first (i.e., 'left') distribution.
            (Default: 'b')
        color_r (str): Color of second (i.e., 'right') distribution.
            (Default: 'r')
        label_l (str): Legend-label of first (i.e., 'left') distribution.
            (Default: '1')
        label_r (str): Legend-label of second (i.e., 'right') distribution.
            (Default: '2')
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)  
        panel_label (None or str): Label for the panel.
            If None, no panel labels is displayed
            (Default: None)
        panel_label_rel_xy (2-tuple of floats): Relative xy-position of the
            panel label.
            (Default: [0.95, 0.95])
        panel_label_fontweight (None or float): Which fontweight to use for the 
            panel label.
            If None, use default font weight for panel label.
            (Default: None)
        x_label (None or str): Label for x-axis. 
            If None, do not display x-axis label.
            (Default: None)
        x_ticks (None or ArrayLike): x-axis tick values.
            If None, do not show x-axis ticks.
            (Default: None)
        show_y_ticks (bool): Should the y-axis ticks be displayed?
            (Default: True)
        fs_dict (dict): Dictionary holding the different fontsizes
            with keys: ['axis', 'leg', 'ticks', 'title'].
            (Default: {'axis': 20, 'leg': 17.5, 'ticks': 15, 'title': 22.5})
        show_legend (bool): Should the legend be displayed?
            (Default: True)
        leg_loc (str): Legend location.
            (Default: 'best')
        leg_handletextpad (None or float): The pad between the legend handle 
            and text, in font-size units.
            If None, use the default padding.
            (Default: None)
        leg_borderpad (None or float): The fractional whitespace inside the 
            legend border, in font-size units.
            If None, use default fractional whitespace.
            (Default: None)
        leg_framealpha (None or float):
            The alpha transparency of the legend's background. If shadow is 
            activated and framealpha is None, the default value is ignored.
            (Default: None)
        leg_vertical_labelspacing (None or float): The vertical space between 
            the legend entries, in font-size units.
            If None, use the default vertical label spacing.
            (Default: None)
        leg_handlelength (None of float): The length of the legend handles, 
            in font-size units.
            If None, use the default legend-handle length.
            (Default: None)
        **kwargs: Keyword-arguments forwarded to 'matplotlib.pyplot.hist' function.

    Return:
        (float): Bin-width*max-histogram-bar-height.
        (object): Panel label object.
    
    """
    if ax is None:
        ax = plt.gca()

    # Bin the data to obtain bars
    _hist_l = ax.hist(x_l, bins=bin_edges, color=color_l, label=label_l, density=True, **kwargs)
    _hist_r = ax.hist(x_r, bins=bin_edges, color=color_r, label=label_r, density=True, **kwargs)

    ###################################################
    ### Set plot specs
    ###################################################
    if x_label is None:
        ax.set_xlabel('')
    else:
        ax.set_xlabel(x_label, fontsize=fs_dict['axis'])

    if x_ticks is None:
        ax.set_xticks([])
    else:
        ax.set_xticks(x_ticks)

    ax.set_ylabel('')
    if show_y_ticks:
        ax.set_yticks([0, 1])
    else:
        ax.set_yticks([])

    ax.tick_params(axis='both', labelsize=fs_dict['ticks'])
    
    # Set x limits
    x_lims   = [min(bin_edges), max(bin_edges)]
    ax.set_xlim(x_lims)

    # Add a panel label if requested
    if panel_label is not None:
        y_lims = [0, 1]
        panel_label_x = x_lims[0]+(x_lims[1]-x_lims[0])*panel_label_rel_xy[0]
        panel_label_y = y_lims[0]+(y_lims[1]-y_lims[0])*panel_label_rel_xy[1]
        panel_label_handle = ax.text(panel_label_x, panel_label_y, panel_label, fontsize=fs_dict['text'], ha='left', va='top', fontweight=panel_label_fontweight)
    else:
        panel_label_handle = None

    # Show the legend if requested
    if show_legend:
        ax.legend(fontsize=fs_dict['leg'], loc=leg_loc, handletextpad=leg_handletextpad, borderpad=leg_borderpad, labelspacing=leg_vertical_labelspacing, handlelength=leg_handlelength, framealpha=leg_framealpha)

    # Determine the maximal histogram entry over both populations,
    # determine the bin-width, and determine the maximal y-value 
    # in the plot from both of these
    max_hist  = max([_hist_l[0].max(), _hist_r[0].max()])
    bin_width = float(np.unique(np.diff(bin_edges))[0])
    max_y     = max_hist*bin_width

    return max_y, panel_label_handle

def plot_tanimoto_similarity_distr(x_l:ArrayLike, 
                                   x_r:ArrayLike, 
                                   bin_edges:ArrayLike,
                                   color_l:str='b',
                                   color_r:str='r',
                                   label_l:str='1',
                                   label_r:str='2',
                                   ax:Optional[object]=None, 
                                   panel_label:Optional[str]=None,
                                   panel_label_rel_xy:Tuple[float, float]=[0.95, 0.95],
                                   panel_label_ha:str='left',
                                   panel_label_fontweight:Optional[float]=None,
                                   x_label:Optional[str]=None,
                                   x_ticks:Optional[ArrayLike]=None,
                                   x_tick_labels:Optional[ArrayLike]=None,
                                   show_y_ticks:bool=True,
                                   fs_dict:dict={'axis': 20, 'leg': 17.5, 'ticks': 15, 'title': 22.5},
                                   show_legend:bool=True,
                                   leg_loc:str='best',
                                   leg_handletextpad:Optional[float]=None,
                                   leg_borderpad:Optional[float]=None,
                                   **kwargs) -> None:
    """
    Make a plot to compare two different pairwise similarity distributions.

    Args:
        x_l (ArrayLike): First (i.e., 'left') distribution.
        x_r (ArrayLike): Second (i.e., 'right') distribution.
        bin_edges (ArrayLike): Bin edges for the histograms of
            created for both distributions.
        color_l (str): Color of first (i.e., 'left') distribution.
            (Default: 'b')
        color_r (str): Color of second (i.e., 'right') distribution.
            (Default: 'r')
        label_l (str): Legend-label of first (i.e., 'left') distribution.
            (Default: '1')
        label_r (str): Legend-label of second (i.e., 'right') distribution.
            (Default: '2')
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)  
        panel_label (None or str): Label for the panel.
            If None, no panel labels is displayed
            (Default: None)
        panel_label_rel_xy (2-tuple of floats): Relative xy-position of the
            panel label.
            (Default: [0.95, 0.95])
        panel_label_ha (str): Horizontal alignment of panel label.
            (Default: 'left')
        panel_label_fontweight (None or float): Which fontweight to use for the 
            panel label.
            If None, use default font weight for panel label.
            (Default: None)
        x_label (None or str): Label for x-axis. 
            If None, do not display x-axis label.
            (Default: None)
        x_ticks (None or ArrayLike): x-axis tick values.
            If None, do not show x-axis ticks.
            (Default: None)
        x_tick_labels (None or ArrayLike): x-axis tick labels.
            If None, show the x-axis tick values (i.e., 'x_ticks') 
            as labels.
            (Default: None)
        show_y_ticks (bool): Should the y-axis ticks be displayed?
            (Default: True)
        fs_dict (dict): Dictionary holding the different fontsizes
            with keys: ['axis', 'leg', 'ticks', 'title'].
            (Default: {'axis': 20, 'leg': 17.5, 'ticks': 15, 'title': 22.5})
        show_legend (bool): Should the legend be displayed?
            (Default: True)
        leg_loc (str): Legend location.
            (Default: 'best')
        leg_handletextpad (None or float): The pad between the legend handle 
            and text, in font-size units.
            If None, use the default padding.
            (Default: None)
        leg_borderpad (None or float): The fractional whitespace inside the 
            legend border, in font-size units.
            If None, use default fractional whitespace.
            (Default: None)
        **kwargs: Keyword-arguments forwarded to 'matplotlib.pyplot.hist' function.

    """
    if ax is None:
        ax = plt.gca()

    # Bin the data to obtain bars
    _hist_l = ax.hist(x_l, bins=bin_edges, color=color_l, label=label_l, density=True, zorder=1, **kwargs)
    _hist_r = ax.hist(x_r, bins=bin_edges, color=color_r, label=label_r, density=True, zorder=0, **kwargs)

    ###################################################
    ### Set plot specs
    ###################################################
    if x_label is None:
        ax.set_xlabel('')
    else:
        ax.set_xlabel(x_label, fontsize=fs_dict['axis'])

    if x_ticks is None:
        ax.set_xticks([])
    else:
        if x_tick_labels is None:
            x_tick_labels = [str(x_tick) for x_tick in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)

    ax.set_ylabel('')
    if show_y_ticks:
        ax.set_yticks([0, 1])
    else:
        ax.set_yticks([])

    ax.tick_params(axis='both', labelsize=fs_dict['ticks'])
    
    # Set x limits
    x_lims   = [min(bin_edges), max(bin_edges)]
    ax.set_xlim(x_lims)

    # Determine the maximal histogram entry over both populations,
    # determine the bin-width, and determine the maximal y-value 
    # in the plot from both of these
    max_y  = max([_hist_l[0].max(), _hist_r[0].max()])*1.05
    y_lims = [0, max_y]
    ax.set_ylim(y_lims)

    # Add a panel label if requested
    if panel_label is not None:
        panel_label_x = x_lims[0]+(x_lims[1]-x_lims[0])*panel_label_rel_xy[0]
        panel_label_y = y_lims[0]+(y_lims[1]-y_lims[0])*panel_label_rel_xy[1]
        ax.text(panel_label_x, panel_label_y, panel_label, fontsize=fs_dict['text'], ha=panel_label_ha, va='top', fontweight=panel_label_fontweight)

    # Show the legend if requested
    if show_legend:
        ax.legend(fontsize=fs_dict['leg'], loc=leg_loc, handletextpad=leg_handletextpad, borderpad=leg_borderpad)


def custom_errorbar(x:np.ndarray, 
                    y:np.ndarray, 
                    yerr:np.ndarray, 
                    ax:Optional[object]=None, 
                    tie_width:float=1.0, 
                    **kwargs) -> None:
    """
    Make custom errorbar plot.

    Args:
        x (np.array): x-values.
        y (np.array): y-values.
        y_err (np.array): y-errors.
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None)
        tie_width (float): Width of the errorbar ties.
            (Default: 1.0)
        **kwargs: keyword-arguments forwarded to 'matplotlib.pyplot.plot' function.

    """
    if ax is None:
        ax = plt.gca()
    
    ax.plot([x, x], [y-yerr, y+yerr], **kwargs)
    ax.plot([x-tie_width/2, x+tie_width/2], [y-yerr, y-yerr], **kwargs)
    ax.plot([x-tie_width/2, x+tie_width/2], [y+yerr, y+yerr], **kwargs)

def make_mae_plot(abs_error_distr_dict:dict, 
                  plot_specs:dict, 
                  ax:Optional[object]=None, 
                  y_max:Optional[float]=None, 
                  y_max_scale:float=1.4, 
                  p_val_annotation_offset:float=0.0, 
                  y_label:str='Mean-Absolute-Error', 
                  show_xticklabels:bool=True, 
                  show_yticklabels:bool=True,
                  title:Optional[str]=None, 
                  panel_label:Optional[str]=None, 
                  display_info:bool=False) -> None:
    """
    Make an mean-absolute error (MAE) plot where the MAEs are
    shown as bars for different absolute error distributions.

    Args:
        abs_error_distr_dict (dict): Absolute error distributions
            of as dictionary of the form
            {<key>: <absolute-error-distribution>, ...}
        plot_specs (dict): Plot specifications as dictionary.
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None) 
        y_max (None or float): Optional y-values maximum.
            If None, this will be determined from the MAEs.
            (Default: None)
        y_max_scale (float): Scale that is applied to y_max
            to obtain the y-axis upper limit.
            (Default: 1.4)
        p_val_annotation_offset (float): (Vertical) offset of the 
            p-value annotations between the different MAE bars.
            (Default: 0.0)
        y_label (str): Label for y-axis
            (Default: 'Mean-Absolute-Error') 
        show_xticklabels (bool):
            Should the x-axis tick labels be shown?
            (Default: True)
        show_yticklabels (bool):
            Should the y-axis tick labels be shown?
            (Default: True)
        title (None or str): To be displayed title.
            If None, no title is shown.
            (Default: None)
        panel_label (None or str): Label for the panel.
            If None, no panel label is shown.
            (Default: None)
        display_info (bool): Display additional information
            to the user.
            (Default: False)
        
    """
    make_fig = False
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        make_fig = True

    suffix_list = plot_specs['suffix_list']
    suffix_position_map = {suffix: position for position, suffix in enumerate(suffix_list)}
    position_suffix_map = {position: suffix for suffix, position in suffix_position_map.items()}
    suffix_label_map    = plot_specs['suffix_label_map']
    suffix_color_map    = plot_specs['suffix_color_map']
    p_val_annotation_color = plot_specs['p_val_annotation_color']

    # Step 1: Do a two-sided Mann-Whitney-U test for all suffix combinations
    p_val_dict = dict()
    for index1, suffix1 in enumerate(suffix_list):
        for index2 in range(index1, len(suffix_list)): # Do suffix-self test
            suffix2 = suffix_list[index2]
            distr1 = abs_error_distr_dict[suffix1]
            distr2 = abs_error_distr_dict[suffix2]

            if (distr1 is None) or (distr2 is None):
                continue

            # Remark: We need method='asymptotic' because we have ties
            _, p_val_12 = stats.mannwhitneyu(distr1, distr2, use_continuity=False, alternative='two-sided', method='asymptotic')
            _, p_val_21 = stats.mannwhitneyu(distr2, distr1, use_continuity=False, alternative='two-sided', method='asymptotic')
            if p_val_12!=p_val_21:
                raise ValueError(f"p-values not symmetric ({p_val_12} and {p_val_21}) for {suffix1}/{suffix2}")
            
            if suffix1==suffix2:
                if p_val_12!=1:
                    raise ValueError(f"Testing difference of the 'absolute error' distribution of '{suffix1}' with itself does lead to p_value={p_val_12} that is not equal 1!!!")

            if display_info:
                print(f"{suffix1}={suffix2} | p-val={p_val_12:.4f} | significant-difference? {p_val_12<0.05}")

            # p-value is symmetric as checked above
            p_val_dict[f"{suffix1}/{suffix2}"] = p_val_12
            p_val_dict[f"{suffix2}/{suffix1}"] = p_val_12

    # Step 2: Order suffix based on bar height/MAE values (in decreasig order)
    suffix_mae_val_tuples = list()
    for suffix in suffix_list:
        abs_error_distr = abs_error_distr_dict[suffix]
        if abs_error_distr is None:
            continue
        
        mae_val = np.mean(abs_error_distr)
        suffix_mae_val_tuples.append((suffix, mae_val))
    
    suffix_mae_val_tuples.sort(key=lambda x: x[1] if x[1] is not None else float('inf'), reverse=True)
    ordered_suffix_list = [suffix for suffix, _ in suffix_mae_val_tuples]
    
    # Step 3: Make the plot
    if title is not None:
        ax.set_title(title, fontsize=plot_specs['title_fs'])

    mae_vals = list()
    #for index, suffix in enumerate(ordered_suffix_list):
    for suffix in suffix_list:
        abs_error_distr = abs_error_distr_dict[suffix]
        if abs_error_distr is None:
            continue
        
        color = suffix_color_map[suffix]
        pos = suffix_position_map[suffix]

        mae_val = np.mean(abs_error_distr)
        mae_err = np.std(abs_error_distr)/math.sqrt(len(abs_error_distr)) # Standard error of mean estimation
        ax.bar(pos, mae_val, color=color, alpha=plot_specs['bar_alpha'], width=plot_specs['bar_width'])
        custom_errorbar(pos, mae_val, yerr=mae_err, ax=ax, tie_width=0.1, color='k', lw=1)

        mae_vals.append(mae_val)

    # Determine the maximal y value for the limits
    if y_max is None:
        y_max = np.max(mae_vals)*y_max_scale

    # Annotate the p-values
    for index1, suffix1 in enumerate(ordered_suffix_list):
        offset_index = 0
        for index2 in range(index1+1, len(ordered_suffix_list)):
            suffix2 = ordered_suffix_list[index2]        

            abs_error_distr1 = abs_error_distr_dict[suffix1]
            if abs_error_distr1 is None:
                continue
            mae_val1 = np.mean(abs_error_distr1)
            abs_error_distr2 = abs_error_distr_dict[suffix2]
            if abs_error_distr2 is None:
                continue
            mae_val2 = np.mean(abs_error_distr2)
            position1 = suffix_position_map[suffix1]
            position2 = suffix_position_map[suffix2]

            p_val = p_val_dict[f"{suffix1}/{suffix2}"]

            p_val_annotation(position1, 
                             position2, 
                             mae_val1, 
                             mae_val2, 
                             p_val, 
                             annotation_style=plot_specs['p_val_annotation_style'], 
                             offset_index=offset_index, 
                             global_offset=p_val_annotation_offset,
                             color=p_val_annotation_color, 
                             y_max=y_max, 
                             ax=ax,
                             fontsize=plot_specs['p_val_annotation_fs'])

            offset_index += 1

    # x-axis
    ax.set_xlim([-0.5, len(suffix_list)-0.5])
    ax.set_xticks([index for index in range(len(suffix_list))])
    if show_xticklabels:
        xticklabels = [suffix_label_map[position_suffix_map[position]] for position in range(len(suffix_list))]
        #ax.set_xticklabels(xticklabels, rotation='vertical', fontsize=plot_specs['xticklabels_fs'])
        ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=plot_specs['xticklabels_fs'])
    else:
        ax.set_xticklabels(['' for _ in suffix_list])

    # y-axis
    ax.tick_params(axis='y', labelsize=plot_specs['yticklabels_fs'])
    ax.set_ylabel(y_label, fontsize=plot_specs['yaxislabel_fs'])
    ax.set_ylim([0, y_max])

    if show_yticklabels:
        yticks = [tick for tick in ax.get_yticks() if tick<=y_max]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in yticks])
    else:
        ax.set_yticklabels([])

    # Plot the panel label
    if panel_label is not None:
        panel_label_x = -0.5+(len(ordered_suffix_list)+1)*0.025
        panel_label_y = y_max*0.975
        ax.text(panel_label_x, panel_label_y, panel_label, fontsize=plot_specs['panel_label_fs'], va='top', ha='left', fontweight=plot_specs['panel_label_fw'])

    if make_fig:
        plt.show()

def p_val_annotation(pos1:float, 
                     pos2:float, 
                     height1:float, 
                     height2:float, 
                     p_val:float, 
                     annotation_style:str='single_level',
                     offset_index:float=0.0, 
                     global_offset:float=0.0, 
                     lw:float=1.0, 
                     color:str='k', 
                     y_max:float=1.0, 
                     ax:Optional[object]=None, 
                     **kwargs) -> None:
    """ 
    Annotate the p-value to two bars within the 'make_mae_plot' function.

    Args:
        pos1 (float): Position of bar 1. 
        pos2 (float): Position of bar 2.
        height1 (float): Height of bar 1. 
        height2 (float): Height of bar 2.
        p_val (float): p-value
        annotation_style (str): p-value annotation style with the options:
            - 'single_level': '*'   if p-value<=0.05
            - 'multi_level':  '*'   if 0.01<p-value<=0.05
                              '**'  if 0.001<p-value<=0.01
                              '***' if p-value<=0.001
            In both annotation styles, 'n.s.' will be annotated in case
            that 0.05<p-value.
            (Default: 'single_level')
        offset_index (int): As the larger height of the two bars
            is used as reference height for the p-value annotation, 
            it can happen (for more than 2 bars) that p-value 
            annotations overlap. To counteract this, one can pass
            a different (integer) offset-index per pair of bars.
            (Default: 0)
        global_offset (float): Global offset of the p-value annotation.
            (Default: 0.0)
        lw (float): Linewidth of the p-value annotation line.
            (Default: 1.0)
        color (str): Color of the p-value annotation.
            (Default: 'k')
        y_max (float): Maximal y-value of the figure (i.e., y_max=y_lims[1]).
            (Deaulft: 1.0)
        ax (None or object): Axis object.
            If None, no axis object is specified (and it is internally created).
            (Default: None) 
        **kwargs: Keyword-arguments forwarded to 'matplotlib.pyplot.text' function.

    """
    if ax is None:
        ax = plt.gca()

    if 0.05<p_val:
        label = 'n.s.'
    else:
        if annotation_style=='single_level':
            label = '*'
        elif annotation_style=='multi_level':
            label = ''
            p_val_bounds = [0.05, 0.01, 0.001]
            for p_val_bound in p_val_bounds:
                if p_val<=p_val_bound:
                    label += '*'
        else:
            err_msg = f"The annotation style '{annotation_style}' is not allowed. Use one of the following: '{single_level}' or 'multi_level'."
            raise ValueError(err_msg)

    mid    = (pos1+pos2)/2
    height = max(height1, height2)+y_max/15*(1+offset_index)+global_offset

    ax.plot([pos1, pos2], [height, height], color=color, lw=lw)
    if label=='n.s.':
        va = 'center'
        dh = y_max/40
    else:
        va = 'center'
        dh = y_max/150

    ax.text(mid, height+dh, label, ha='center', va=va, color=color, **kwargs)