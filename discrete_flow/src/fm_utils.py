import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable, Literal
from einops import rearrange, repeat
import time
from collections import defaultdict
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

EPS = 1e-10
gumbel_distribution = torch.distributions.gumbel.Gumbel(torch.tensor([0.0]), torch.tensor([1.0]))

GUMBEL_MEAN = gumbel_distribution.mean
GUMBEL_STD = (gumbel_distribution.variance)**0.5

def flow_matching_sampling(
    num_samples: int,
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    S: int,
    D: int,
    device: torch.device,
    dt: float = 0.001,
    mask_idx: Optional[int] = None,
    pad_idx: Optional[int] = None,
    predictor_log_prob: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    cond_denoising_model: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    guide_temp: float = 1.0,
    stochasticity: float = 0,
    use_tag: bool = False,
    batch_size: int = 500,
    argmax_final: bool = True,
    max_t: float = 1.0,
    x1_temp: float = 1.0,
    do_purity_sampling: bool = False,
    purity_temp: float = 1.0,
    num_unpadded_freq_dict: Optional[dict[int, float]] = None,
    eps: float = 1e-9,
    ##############################
    ## IMPORTANT PARAMETER
    guidance_at_x1: bool = False, ## if True, use guidance at x1

    ##############################
    ### Parameters for gradient-based guidance (calculate the guided rate matrix)
    ###############################
    ### A. Parameters for guidance at x1
    only_sample_for_rt_star: bool = False,
    only_sample_for_unmasking: bool = False,
    sample_k_for_guided_matrix: int = 10,
    log_Rt_ratio_temp: float = 1.0,
    log_Rt_ratio_cutoff: float = 80.0,

    ### B. Parameters for guidance at xt
    ### Basic Parameters for estimating the p(y|xt) with time-independent predictor p(y|x1) ###
    predict_on_x1: bool = True,
    sample_k_for_prob_xt_estimation: int = 10,
    low_threshold: float = 1e-8,

    #### gradient-based guidance ####
    use_grad_fn_v1: bool = False,
    gumbel_softmax_t: float = 1.0,

    ##### + Gumbel-max based sampling + rescale ####
    sample_xt_with_gumbel_max: bool = False,
    gamma1: float = 1.0,
    gamma2: float = 1.0,
    gumbel_norm_expectation: float = None,
    ## rescale
    strength_rescale: bool = False,
    strength_rescale_after_combination: bool = False,
    add_gumbel_mean: bool = False,
    rescale_method: str = "new",

    ###############################
    ### A/B. Parameters for MCTS (both for guidance at x1 and xt)
    mcts: bool = False,
    branch_out_size: int = 1,
    active_set_size: int = 1,
    ##
    ## For MCTS at xt, whether to use guided rate matrix for sampling and select top k samples based on the predictors
    use_guided_rate: bool = True,

    ###### [Rebuttal] Parameters Added ######
    ### guidance at xt, MCTS sampling
    svdd: bool = False,  # whether to use SVDD for the rate matrix
    svdd_temp: float = 1.0,  # temperature for SVDD
    tds: bool = False,  # whether to use the Twisted Diffusion Sampler (TDS) method
    tds_return_all: bool = False,  # whether to return all samples for TDS
    guidance_start_step: int = 0,  # the step to start applying guidance
    guidance_end_step: int = -1,  # the step to stop applying guidance, -1 means until the end
    ###### [After Rebuttal] Parameters Added ######
    tds_reweight_temp: float = 1.0,  # temperature for TDS reweighting
    tds_prob_ratio_temp: float = 1.0,  # temperature for TDS probability ratio
    add_remask_rate: bool = True,  # whether to add remask rate
    ########################################
    # [Only for Debugging]
    without_guidance: bool = False,
    sample_max: bool = False,
    verbose: bool = False,
):
    """
    Generates samples using flow matching with optional predictor or predictor-free guidance.
    This is a wrapper function that generates samples in batches for memory efficiency.

    Args:
        num_samples (int): Total number of samples to generate
        denoising_model (nn.Module): The unconditional denoising model
        S (int): Size of the categorical state space
        D (int): Dimensionality of each sample
        device (torch.device): Device to run generation on
        dt (float, optional): Time step size for Euler integration. Defaults to 0.001.
        mask_idx (int, optional): Index used for mask token. If None, uses S-1.
        pad_idx (int, optional): Index used for padding token. If None, no padding is used.
        predictor_log_prob (callable, optional): Function that takes (x, t) and returns log p(y|x,t) for predictor guidance
        cond_denoising_model (callable, optional): Function that takes (x, t) and returns logits for predictor-free guidance
        guide_temp (float, optional): Guidance temperature (1/ \gamma). Lower temperature = stronger guidance. Defaults to 1.0.
        stochasticity (float, optional): Amount of stochastic noise in sampling. Defaults to 0.
        use_tag (bool, optional): Whether to use Taylor-approximated guidance. Defaults to False.
        batch_size (int, optional): Batch size for generation. Defaults to 500.
        argmax_final (bool, optional): Whether to argmax final outputs. Defaults to True.
        max_t (float, optional): Maximum time value for sampling. Defaults to 1.0.
        x1_temp (float, optional): Temperature for x1 prediction logits. Defaults to 1.0.
        do_purity_sampling (bool, optional): Whether to use purity-based sampling. Defaults to False.
        purity_temp (float, optional): Temperature for purity sampling. Defaults to 1.0.
        num_unpadded_freq_dict (dict, optional): Dictionary of frequencies for number of unpadded tokens
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-9.

        ##############################
        For the parameters in different guidance methods, please check the detailed description in the function

    Returns:
        np.ndarray: Generated samples of shape (num_samples, D)
    """
    # Check if using predictor guidance
    use_predictor_guidance = predictor_log_prob is not None
    # Check if using predictor-free guidance
    use_predictor_free_guidance = cond_denoising_model is not None
    print(
        f"Generating {num_samples} samples: dt={dt}, "
        f"stochasticity={stochasticity}, "
        f"guide_temp={guide_temp}, "
        f"predictor_guidance={use_predictor_guidance}, "
        f"predictor_free_guidance={use_predictor_free_guidance}, "
        f"use_tag={use_tag}"
    )
    # Adjust batch size if needed
    if batch_size > num_samples:
        batch_size = num_samples

    # Generate samples in batches
    counter = 0
    samples = []
    time_info_dict = {}
    if verbose:
        torch.cuda.synchronize()
        start_time = time.perf_counter()    

    while True:
        if guidance_at_x1:
            ### ADD GUIDANCE AT X1
            x1, exp_time_info_dict = x1_weighted_flow_matching_sampling_masking_euler(
                denoising_model=denoising_model,
                batch_size=batch_size,
                S=S,
                D=D,
                device=device,
                dt=dt,
                mask_idx=mask_idx,
                pad_idx=pad_idx,
                predictor_log_prob=predictor_log_prob,
                stochasticity=stochasticity,
                argmax_final=argmax_final,
                max_t=max_t,
                x1_temp=x1_temp,
                num_unpadded_freq_dict=num_unpadded_freq_dict,
                ##############################
                ### IMPORTANT PARAMETER
                guide_temp=guide_temp,
                ## how to compute the rate matrix
                only_sample_for_rt_star=only_sample_for_rt_star,
                only_sample_for_unmasking=only_sample_for_unmasking,
                sample_k_for_guided_matrix=sample_k_for_guided_matrix,
                log_Rt_ratio_temp=log_Rt_ratio_temp,
                log_Rt_ratio_cutoff=log_Rt_ratio_cutoff,
                ### mcts
                mcts=mcts,
                branch_out_size=branch_out_size,
                active_set_size=active_set_size,
                ### only for debugging
                without_guidance=without_guidance,
                verbose=verbose,
                ###### added after rebuttal ######
                svdd=svdd,
                svdd_temp=svdd_temp,  # temperature for SVDD
                add_remask_rate=add_remask_rate,  # whether to add remask rate
            )
        else:
            ### ADD GUIDANCE AT XT
            x1, exp_time_info_dict  = flow_matching_sampling_masking_euler(
                denoising_model=denoising_model,
                batch_size=batch_size,
                S=S,
                D=D,
                device=device,
                dt=dt,
                mask_idx=mask_idx,
                pad_idx=pad_idx,
                predictor_log_prob=predictor_log_prob,
                cond_denoising_model=cond_denoising_model,
                stochasticity=stochasticity,
                use_tag=use_tag,
                argmax_final=argmax_final,
                max_t=max_t,
                x1_temp=x1_temp,
                do_purity_sampling=do_purity_sampling,
                purity_temp=purity_temp,
                num_unpadded_freq_dict=num_unpadded_freq_dict,
                eps=eps,
                ########################################
                guide_temp=guide_temp,

                ##############################
                ### Paramters for MCTS
                mcts=mcts,
                branch_out_size=branch_out_size,
                active_set_size=active_set_size,
                use_guided_rate=use_guided_rate, 
                ##############################
                #### Basic Parameters for estimating p(y|xt) with time-independent predictor p(y|x1) ###
                predict_on_x1=predict_on_x1,
                sample_k_for_prob_xt_estimation=sample_k_for_prob_xt_estimation,
                low_threshold=low_threshold,

                ### Parameters for training-free gradient-based guidance
                use_grad_fn_v1=use_grad_fn_v1,
                gumbel_softmax_t=gumbel_softmax_t,

                ### + Parameters for Gumbel-max based sampling + rescale 
                sample_xt_with_gumbel_max=sample_xt_with_gumbel_max,
                gumbel_norm_expectation=gumbel_norm_expectation,
                gamma1=gamma1,
                gamma2=gamma2,
                strength_rescale=strength_rescale,
                strength_rescale_after_combination=strength_rescale_after_combination,
                add_gumbel_mean=add_gumbel_mean,
                rescale_method=rescale_method,

                ##### parameters added during the rebuttal period #####
                svdd=svdd,
                svdd_temp=svdd_temp,
                tds=tds,  # whether to use the Twisted Diffusion Sampler (TDS) method
                tds_return_all=tds_return_all,  # whether to return all samples for TDS
                guidance_start_step=guidance_start_step,  # the step to start applying guidance
                guidance_end_step=guidance_end_step,  # the step to stop applying guidance, -1 means until the end
                ### parameters added after the rebuttal period
                tds_reweight_temp=tds_reweight_temp,  # temperature for TDS reweighting
                tds_prob_ratio_temp=tds_prob_ratio_temp,  # temperature for TDS probability ratio
                ########################################
                ### [Only for Debugging]
                sample_max=sample_max,
                verbose=verbose,
            )
        samples.append(x1)
        # counter += batch_size
        counter += len(x1)
        if tds and tds_return_all:
            num_samples = num_samples * active_set_size
        print(f"{counter} out of {num_samples} generated")
        if counter >= num_samples:
            break

    if verbose:
        torch.cuda.synchronize()
        batch_inference_time = time.perf_counter()-start_time ## inference time for one batch
        start_time = time.perf_counter()
        sample_inference_time = batch_inference_time / counter ## inference time for one sample
        time_info_dict["batch_inference_time"] = batch_inference_time
        time_info_dict["sample_inference_time"] = sample_inference_time

        time_info_dict.update(exp_time_info_dict)

    # Concatenate and trim to exact number of samples requested
    samples = np.concatenate(samples, axis=0)[:num_samples]
    return samples, time_info_dict

########################################################################
#### B. GUIDANCE AT X1
## (1) MCTS at x1
## (2） ICLR TFG-Flow ####
########################################################################

## not support classifier-free guidance
@torch.no_grad()
def x1_weighted_flow_matching_sampling_masking_euler(
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: int,
    S: int,
    D: int,
    device: torch.device,
    dt: float = 0.001,
    mask_idx: Optional[int] = None,
    pad_idx: Optional[int] = None,
    predictor_log_prob: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    guide_temp: float = 1.0,
    stochasticity: float = 0,
    argmax_final: bool = True,
    max_t: float = 1.0,
    x1_temp: float = 1.0,
    num_unpadded_freq_dict: Optional[dict[int, float]] = None,
    ##############################
    sample_k_for_guided_matrix: int = 10,
    only_sample_for_rt_star: bool = False,
    only_sample_for_unmasking: bool = False,
    mcts: bool = False,
    active_set_size: int = 1, ## only valid when mcts is True
    branch_out_size: int = 1, ## only valid when mcts is True
    without_guidance: bool = False,
    log_Rt_ratio_temp: float = 1.0,
    log_Rt_ratio_cutoff: float = 80.0,
    verbose: bool = False,
    ###### added after rebuttal ######
    svdd: bool = False,  # whether to use SVDD for the rate matrix
    svdd_temp: float = 1.0,  # temperature for SVDD
    ###### added after neurips rebuttal ######
    add_remask_rate: bool = True,
) -> np.ndarray:
    """
    Generates samples using Euler integration of the discrete flow matching model with optional guidance.

    This function implements the core sampling algorithm for discrete flow matching with masking noise.
    It supports both predictor guidance and predictor-free guidance, and includes options for
    purity-based sampling and padding token handling.

    Args:
        denoising_model: Model that takes (x_t: [B,D], t: [B]) and returns logits [B,D,S]
        batch_size: Number of samples to generate in parallel
        S: Size of categorical state space (vocabulary size)
        D: Dimension of each sample (sequence length)
        device: Device to run generation on
        dt: Time step size for Euler integration
        mask_idx: Token index used for masking. Defaults to S-1
        pad_idx: Optional token index used for padding
        predictor_log_prob: Optional predictor function for guided sampling that takes (x,t)
            and returns log p(y|x,t) of shape [B]
        guide_temp: Temperature for guidance (1 / \gamma). Lower = stronger guidance
        stochasticity: Amount of stochastic noise in sampling
        argmax_final: Whether to use argmax for any remaining masked tokens at end
        max_t: Maximum time value to run sampling
        x1_temp: Temperature for softmax of model logits
        do_purity_sampling: Whether to weight sampling by prediction confidence
        purity_temp: Temperature for purity-based sampling weights
        num_unpadded_freq_dict: Optional dict mapping num unpadded tokens to frequencies
        eps: Small constant for numerical stability

        sample_k_for_guided_matrix (int): the number of sampled x1 for monte carlo estimation of guided rate matrix
        only_sample_for_rt_star (bool): whether to only sample for rt_star
        only_sample_for_unmasking (bool): whether to only sample for unmasking
        mcts (bool): whether to apply MCTS search method
        active_set_size (int): the number of active set
        branch_out_size (int): the number of branch out
        without_guidance (bool): whether to use guidance (only for debugging)
        log_Rt_ratio_temp (float): the temperature for the ratio between the conditional and unconditional rate matrix
        log_Rt_ratio_cutoff (float): the cutoff for the ratio between the conditional and unconditional rate matrix
        verbose (bool): whether to store the time information

    Returns:
        numpy.ndarray: Generated samples of shape [batch_size, D]
    """
    if not mcts:
        assert active_set_size == 1, "active_set_size must be 1 if not using MCTS"
        assert branch_out_size == 1, "branch_out_size must be 1 if not using MCTS"

    if mask_idx is None:
        mask_idx = S - 1

    B = batch_size

    # Sample initial xt
    xt = mask_idx * torch.ones((B, D), dtype=torch.long, device=device)

    t = 0.0
    num_steps = int(1 / dt)  # TODO: Use ceil or int?
    mask_one_hot = torch.zeros((S,), device=device)
    mask_one_hot[mask_idx] = 1.0

    # Treat the case where fixed pads should be used
    pad_mask = None
    if pad_idx is not None:
        pad_one_hot = torch.zeros((S,), device=device)
        pad_one_hot[pad_idx] = 1.0

        # If 'num_unpadded_freq_dict' is not None,
        # sample pads for x0 (=xt at time t=0) and pad xt
        # overwriting the current xt
        if num_unpadded_freq_dict is not None:
            xt, pad_mask = sample_pads_for_x0(xt, pad_idx, num_unpadded_freq_dict)
    
    xt = xt.unsqueeze(1).repeat(1, active_set_size, 1)
    if pad_mask is not None:
        pad_mask = pad_mask.unsqueeze(1).repeat(1, active_set_size, 1)


    # R_t_diff_norm_list = []
    if mcts:
        sample_k = branch_out_size
    else:
        sample_k = sample_k_for_guided_matrix

    sel_top_num = active_set_size
    all_time_info = defaultdict(list)
    if verbose:
        torch.cuda.synchronize()
        initial_time_info = time.perf_counter()

    valid_steps = 0
    for step_idx in tqdm(range(num_steps)):
        # Get p(x1 | xt), scaled by temperature
        # This is the unconditional prediction
        # If denoising model trained unconditionally, it doesn't use the cls input
        # If it is trained conditionally, this is the index of the unconditional class

        B_origianl = xt.shape[0]
        xt = rearrange(xt, 'B N D -> (B N) D')
        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, 'B N D -> (B N) D')
        B = xt.shape[0]

        if verbose:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        logits = denoising_model(xt, t * torch.ones((B,), device=device))  # (B, D, S)
        if verbose:
            torch.cuda.synchronize()
            all_time_info["denoising_model"].append(time.perf_counter()-start_time)

        pt_x1_probs = F.softmax(logits / x1_temp, dim=-1)  # (B, D, S)

        x1_k = torch.distributions.Categorical(probs=repeat(pt_x1_probs, 'bs n atomnames -> bs k n atomnames', k=sample_k)).sample()
        x1_k = rearrange(x1_k, 'B K D -> (B K) D')
        # if pad_mask is not None and pad_idx is not None:
        #     x1_k[rearrange(repeat(pad_mask, 'B D -> B K D', K=sample_k), 'B K D -> (B K) D')] = pad_idx

        xt_k = repeat(xt, 'B D -> B K D', K=sample_k)
        xt_k = rearrange(xt_k, 'B K D -> (B K) D')

        assert not (only_sample_for_rt_star and only_sample_for_unmasking), "only_sample_for_rt_star and only_sample_for_unmasking cannot be True at the same time"

        if pad_mask is not None and pad_idx is not None: ## TODO: maybe not necessary for the cases other than mcts 
            # x1_k[rearrange(repeat(pad_mask, 'B D -> B K D', K=sample_k), 'B K D -> (B K) D')] = pad_idx

            pad_mask_K = rearrange(repeat(pad_mask, 'B D -> B K D', K=sample_k), 'B K D -> (B K) D')
            x1_k[pad_mask_K] = pad_idx

        if mcts:
            if sample_k != 1 or (active_set_size != 1 and sel_top_num == 1):
                

                log_prob_x1_k = predictor_log_prob(x1_k) ## (B N K) 
                log_prob_x1_k = rearrange(log_prob_x1_k, "(B N_K) -> B N_K", B=B_origianl, N_K=active_set_size * sample_k)
                xt_k = rearrange(xt_k, "(B N_K) D -> B N_K D", B=B_origianl, N_K=active_set_size * sample_k)
                x1_k = rearrange(x1_k, "(B N_K) D -> B N_K D", B=B_origianl, N_K=active_set_size * sample_k)
                if pad_mask is not None:
                    pad_mask = pad_mask_K
                    pad_mask = rearrange(pad_mask, "(B N_K) D -> B N_K D", B=B_origianl, N_K=active_set_size * sample_k)

                if svdd: ## [ADDED ON May 14th, DEBUGGING]
                    # log_y_prob = torch.log(torch.clamp(y_prob, min=low_threshold))
                    ### TODO: check whether the below is necessary
                    full_invalid_mask = (torch.sum(log_prob_x1_k != -torch.inf, -1) == 0) ## Deal with invalid log prob -inf
                    full_invalid_mask = full_invalid_mask[:, None]  # (B, 1)
                    log_prob_x1_k = torch.where(full_invalid_mask, torch.zeros_like(log_prob_x1_k[0]), log_prob_x1_k) 

                    y_prob_weight = torch.softmax(log_prob_x1_k / svdd_temp, dim=-1)
                    # y_prob_weight = torch.clamp(y_prob ** (1 / svdd_temp), min=max((low_threshold ** (1 / svdd_temp), 1e-40)))
                    idx_x1_top_n = torch.multinomial(y_prob_weight, sel_top_num, replacement=False)
                else:
                    log_prob_x1_top_n, idx_x1_top_n = torch.topk(log_prob_x1_k, k=sel_top_num, dim=1, largest=True)
                ## B N

                xt_k = torch.gather(xt_k, dim=1, index=idx_x1_top_n.unsqueeze(-1).repeat(1,1, xt_k.shape[-1]))
                xt_k = rearrange(xt_k, "B N D -> (B N) D")

                x1_k = torch.gather(x1_k, dim=1, index=idx_x1_top_n.unsqueeze(-1).repeat(1,1, x1_k.shape[-1]))
                x1_k = rearrange(x1_k, "B N D -> (B N) D") ## B sel_top_num D

                if pad_mask is not None:
                    pad_mask = torch.gather(pad_mask, dim=1, index=idx_x1_top_n.unsqueeze(-1).repeat(1,1, pad_mask.shape[-1]))
                    pad_mask = rearrange(pad_mask, "B N D -> (B N) D")

            only_sample_for_unmasking = False if add_remask_rate else True
            only_sample_for_rt_star = False

        
        R_t_k = compute_rate_matrix_given_x1(
            xt_k, x1_k, mask_one_hot, mask_idx, S, stochasticity, t, pad_idx, 
            only_sample_for_rt_star = only_sample_for_rt_star,
            only_sample_for_unmasking = only_sample_for_unmasking,
            # pt_x1_probs = pt_x1_probs
        ) ## (B*K, D, S)

        if mcts:
            R_t = R_t_k
            xt = xt_k

        else:
            ## if only_guide_on_rd_star, R_t_k is R_t_d_star
            ## else R_t_k is R_t, R_t_d_star + R_t_db
            R_t_k = rearrange(R_t_k, '(B K) D S -> B K D S', B=B, K=sample_k)

            log_prob_x1_k = predictor_log_prob(x1_k) ## (B N K) 
            log_prob_x1_k = rearrange(log_prob_x1_k, '(bs k) -> bs k', bs=B)

            # if not without_guidance:
            #     energy = torch.softmax(log_prob_x1_k / guide_temp, dim = -1) ## (B, K)
            #     R_t = torch.sum(R_t_k * energy[:, :, None, None], dim = 1) ## (B, D, S)
            # else:
            #     R_t = torch.mean(R_t_k, dim = 1) ## (B, D, S)
            
            ##### UPDATE #####
            ### Support ~predictor-free guidance
            Uncond_R_t = torch.mean(R_t_k, dim = 1) ## (B, D, S)
            if not without_guidance:
                ### TODO: check whether the below is necessary
                full_invalid_mask = (torch.sum(log_prob_x1_k != -torch.inf, -1) == 0) ## Deal with invalid log prob -inf
                full_invalid_mask = full_invalid_mask[:, None]  # (B, 1)
                log_prob_x1_k = torch.where(full_invalid_mask, torch.zeros_like(log_prob_x1_k[0]), log_prob_x1_k) 

                energy = torch.softmax(log_prob_x1_k / guide_temp, dim = -1) ## (B, K)
                Cond_R_t = torch.sum(R_t_k * energy[:, :, None, None], dim = 1) ## (B, D, S)
                # R_t = ((Cond_R_t+EPS) ** predictor_free_weight) * ((Uncond_R_t+EPS) ** (1 - predictor_free_weight))
                assert log_Rt_ratio_temp != 0, "log_Rt_ratio_temp cannot be 0"
                log_Rt_ratio = (1 / log_Rt_ratio_temp) * (torch.log(Cond_R_t + EPS) - torch.log(Uncond_R_t + EPS)) 
                log_Rt_ratio = torch.clamp(log_Rt_ratio, max=log_Rt_ratio_cutoff)
                R_t = torch.exp(log_Rt_ratio) * Uncond_R_t

            else:
                R_t = Uncond_R_t

            if only_sample_for_rt_star:   
                R_t_db = compute_rate_matrix(
                    xt, pt_x1_probs, mask_idx, mask_one_hot, pad_one_hot, 
                    stochasticity, t, pad_idx,
                    add_rt_star=False
                )
                R_t = R_t + R_t_db

            elif only_sample_for_unmasking and add_remask_rate:
                R_t_remask = compute_rate_matrix_remask(xt, mask_idx, mask_one_hot, stochasticity, pad_idx)
                R_t = R_t + R_t_remask


        # Set the diagonal of the rates to negative row sum
        R_t.scatter_(-1, xt[:, :, None], 0.0)
        R_t.scatter_(-1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))

        # Obtain probabilities from the rates
        step_probs = (R_t * dt).clamp(min=0.0, max=1.0)
        step_probs.scatter_(-1, xt[:, :, None], 0.0)
        step_probs.scatter_(
            -1,
            xt[:, :, None],
            (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0) ## the sum of step_probs may not be 1.0

        # Sample the next xt
        try:
            xt = torch.distributions.Categorical(step_probs).sample()  # (B, D) ## if the sum of step_probs is not 1.0, it will normalize the step_probs
            # if mcts:
            xt = rearrange(xt, "(B N) D -> B N D", B=B_origianl, N=sel_top_num)
            if pad_mask is not None:
                pad_mask = rearrange(pad_mask, "(B N) D -> B N D", B=B_origianl, N=sel_top_num)
            B = xt.shape[0]
        except ValueError:
            raise ValueError(
                "ValueError in 'torch.distributions.Categorical(step_probs).sample()', step_probs might not valid."
            )
        # print(f"xt: {xt.unique()}")
        t += dt
        valid_steps += 1

        if mcts and t + dt > max_t:
            sel_top_num = 1

        if t > max_t:
            xt = xt.squeeze(1)
            break
    
    if verbose and len(all_time_info) > 0:
        torch.cuda.synchronize()
        total_time = time.perf_counter() - initial_time_info
        # assert step_idx == num_steps - 1, "step_idx should be equal to num_steps - 1"
        all_time_info["batch_inference_time_per_step"] = total_time / valid_steps
        all_time_info["sample_inference_time_per_step"] = total_time / (batch_size * valid_steps)

        denoising_model_time = all_time_info.pop("denoising_model")
        all_time_info["batch_denoising_model_time"] = np.sum(denoising_model_time)
        all_time_info["batch_denoising_model_time_per_step"] = np.mean(denoising_model_time)
        all_time_info["sample_denoising_model_time_per_step"] = np.mean(denoising_model_time) / batch_size
        all_time_info["batch_size"] = batch_size
        all_time_info["valid_steps"] = valid_steps

    # For any state that is still masked, take the argmax of the logits
    # of the final xt
    if argmax_final:
        xt_is_mask = (xt == mask_idx).view(B, D).float()
        logits = denoising_model(xt, t * torch.ones((B,), device=device))  # (B, D, S)
        xt = torch.argmax(logits, dim=-1) * xt_is_mask + xt * (1 - xt_is_mask)
    
    return xt.detach().cpu().numpy(), all_time_info


def compute_rate_matrix_given_x1(
        xt, x1, mask_one_hot, mask_idx,
        S, stochasticity, t,
        pad_idx = None,
        only_sample_for_rt_star = False,
        only_sample_for_unmasking = False,
        # pt_x1_probs = None, 
        ):
    '''
        Compute the rate matrix R(x_tilde, xt | x1). Reference: D.4 in the paper
    '''
    assert not (only_sample_for_rt_star and only_sample_for_unmasking), "only_sample_for_rt_star and only_sample_for_unmasking cannot be True at the same time"
    B, D = xt.shape[:2]
    xt_is_mask = (xt == mask_idx).view(B, D, 1).float()
    x1_one_hot = F.one_hot(x1, num_classes=S).float() ## (B, D, S)    

    R_t_d_star = xt_is_mask * x1_one_hot * (1 / (1 - t))
    if pad_idx is not None:
        x1_is_pad = (x1 == pad_idx).view(B, D, 1).float()
        R_t_d_star *= 1 - x1_is_pad
    
    if only_sample_for_rt_star:
        return R_t_d_star
    
    if only_sample_for_unmasking:
        R_t_add = xt_is_mask * x1_one_hot * (stochasticity * t / (1 - t)) 
        if pad_idx is not None:
            # x1_is_pad = (x1 == pad_idx).view(B, D, 1).float()
            R_t_add *= 1 - x1_is_pad
        R_t_unmask = R_t_d_star + R_t_add
        return R_t_unmask

    xt_is_x1 = (xt == x1).view(B, D, 1).float() ## (B, D, 1)

    R_t_db = stochasticity * (
        xt_is_mask * x1_one_hot * t / (1 - t) +
        xt_is_x1 * mask_one_hot.view(1, 1, -1)
    )  # (B, D, S)

    if pad_idx is not None:
        x1_is_pad = (x1 == pad_idx).view(B, D, 1).float()
        R_t_db *= 1 - x1_is_pad
    
    R_t = R_t_d_star + R_t_db
    return R_t


def compute_rate_matrix(xt, pt_x1_probs, mask_idx, mask_one_hot, pad_one_hot, stochasticity, t, pad_idx = None, add_rt_star = True):
    add_rt_star = int(add_rt_star)
    ### reference: https://arxiv.org/pdf/2402.04997
    B, D = xt.shape[:2]
    xt_is_mask = (xt == mask_idx).view(B, D, 1).float()
    R_t = (
        xt_is_mask * pt_x1_probs * (add_rt_star + stochasticity * t) / (1 - t)
    )  # (B, D, S)

    if pad_idx is not None:
        R_t *= 1 - pad_one_hot.view(1, 1, -1)
    
    remask_rates = (1 - xt_is_mask) * mask_one_hot.view(1, 1, -1) * stochasticity
    ### (1 - xt_is_mask) = p(x1 == xt | xt)
    if pad_idx is not None:
        xt_is_pad = (xt == pad_idx).view(B, D, 1).float()
        remask_rates *= 1 - xt_is_pad

    R_t += remask_rates

    return R_t


def compute_rate_matrix_remask(xt, mask_idx, mask_one_hot, stochasticity, pad_idx = None):
    ### reference: https://arxiv.org/pdf/2402.04997 page 38
    B, D = xt.shape[:2]
    xt_is_mask = (xt == mask_idx).view(B, D, 1).float()

    remask_rates = (1 - xt_is_mask) * mask_one_hot.view(1, 1, -1) * stochasticity

    if pad_idx is not None:
        xt_is_pad = (xt == pad_idx).view(B, D, 1).float()
        remask_rates *= 1 - xt_is_pad

    return remask_rates


########################################################################
#### B. GUIDANCE AT XT
## (1) MCTS at xt
## (2) Time-independent Gradient-based Guidance
########################################################################

@torch.no_grad()
def flow_matching_sampling_masking_euler(
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: int,
    S: int,
    D: int,
    device: torch.device,
    dt: float = 0.001,
    mask_idx: Optional[int] = None,
    pad_idx: Optional[int] = None,
    predictor_log_prob: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    cond_denoising_model: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    guide_temp: float = 1.0,
    stochasticity: float = 0,
    use_tag: bool = False,
    argmax_final: bool = True,
    max_t: float = 1.0,
    x1_temp: float = 1.0,
    do_purity_sampling: bool = False,
    purity_temp: float = 1.0,
    num_unpadded_freq_dict: Optional[dict[int, float]] = None,
    eps: float = 1e-9,

    ### Basic Parameters for estimating p(y|xt) with ###
    predict_on_x1: bool = True,
    sample_k_for_prob_xt_estimation: int = 10,
    low_threshold: float = 1e-8,

    ##############################
    ### Paramters for MCTS
    mcts: bool = False,
    active_set_size: int = 1,
    branch_out_size: int = 1,
    use_guided_rate: bool = True, 
    ##############################
    ### Parameters for training-free gradient-based guidance
    use_grad_fn_v1: bool = False,
    gumbel_softmax_t: float = 1.0,

    ### Additional Parameters for Gumbel-max based sampling + rescale 
    sample_xt_with_gumbel_max: bool = False,
    gamma1: float = 1.0,
    gamma2: float = 1.0,
    strength_rescale: bool = False,
    strength_rescale_after_combination: bool = False,
    rescale_method: str = "new",
    add_gumbel_mean: bool = False,
    gumbel_norm_expectation: float = None,
    #### added during rebuttal ######
    svdd: bool = False,  # whether to use the svdd method
    svdd_temp: float = 1.0,  # the temperature for the svdd method
    tds: bool = False,  # whether to use the Twisted Diffusion Sampler (TDS) method
    tds_return_all: bool = False,  # whether to return all samples for TDS
    guidance_start_step: int = 0,  # the step to start applying guidance
    guidance_end_step: int = -1,  # the step to stop applying guidance, -1 means until the end
    #### added after rebuttal ######
    tds_reweight_temp: float = 1.0,  # the temperature for the TDS reweighting
    tds_prob_ratio_temp: float = 1.0,  # the temperature for the TDS probability ratio
    ########################################
    ### [Only for Debugging]
    sample_max: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Generates samples using Euler integration of the discrete flow matching model with optional guidance.

    This function implements the core sampling algorithm for discrete flow matching with masking noise.
    It supports both predictor guidance and predictor-free guidance, and includes options for
    purity-based sampling and padding token handling.

    Args:
        denoising_model: Model that takes (x_t: [B,D], t: [B]) and returns logits [B,D,S]
        batch_size: Number of samples to generate in parallel
        S: Size of categorical state space (vocabulary size)
        D: Dimension of each sample (sequence length)
        device: Device to run generation on
        dt: Time step size for Euler integration
        mask_idx: Token index used for masking. Defaults to S-1
        pad_idx: Optional token index used for padding
        predictor_log_prob: Optional predictor function for guided sampling that takes (x,t)
            and returns log p(y|x,t) of shape [B]
        cond_denoising_model: Optional conditional model for predictor-free guidance that
            takes (x,t) and returns logits [B,D,S]
        guide_temp: Temperature for guidance (1 / \gamma). Lower = stronger guidance
        stochasticity: Amount of stochastic noise in sampling
        use_tag: Whether to use Taylor approximation for predictor guidance
        argmax_final: Whether to use argmax for any remaining masked tokens at end
        max_t: Maximum time value to run sampling
        x1_temp: Temperature for softmax of model logits
        do_purity_sampling: Whether to weight sampling by prediction confidence
        purity_temp: Temperature for purity-based sampling weights
        num_unpadded_freq_dict: Optional dict mapping num unpadded tokens to frequencies
        eps: Small constant for numerical stability

        ### Basic Parameters for estimating p(y|xt) with ###
        sample_k_for_prob_xt_estimation (int): the number of samples to be sampled for monte carlo estimation
        low_threshold (float): the threshold for the low probability
        ##

        ##############################
        ### Paramters for MCTS
        mcts (bool): whether to apply MCTS search method
        active_set_size (int): the number of active set
        branch_out_size (int): the number of branch out
        use_guided_rate (bool): whether to use guided rate, for the vanilla mcts method, 
            please set this as false to use unguided rate matrix for sampling and select top k samples based on the predictors
            if you want to use guide rate matrix, please set this as True

        ##################################
        ### Parameters for training-free gradient-based guidance
        use_grad_fn_v1 (bool): whether to use the first version of gradient-based guidance 
            (grad_fn_v2 is more accurate than grad_fn_v1)
        gumbel_softmax_t: temperature for gumbel softmax to estimate the sampling process

        ########################################    
        ### Parameters for Gumbel-max based sampling + rescale
        sample_xt_with_gumbel_max (bool): whether to sample xt with gumbel-max
        gamma1: the weight for gumbel noise
        gamma2: the weight for log probability ratio
        strength_rescale: whether to rescale the strength of log probability ratio
        strength_rescale_after_combination: whether to rescale the strength of log probability ratio after combination
        rescale_method: the method for rescaling the strength of the gumbel noise-like features (default: "new")
        add_gumbel_mean: whether to add the mean of the gumbel noise to the rescaled gumbel noise-like features (should not matter)
        gumbel_norm_expectation: the expectation of the gumbel noise (only used for old rescale method)

        #### ADDED DURING REBUTTAL ######
        svdd: whether to use the svdd method (Soft Value-based Decoding in Diffusion models (SVDD))
            https://arxiv.org/pdf/2408.08252v4 Algorithm 1
        svdd_temp: the temperature for the SVDD method (default: 1.0)

        tds: whether to use the Twisted Diffusion Sampler (TDS) method
            https://arxiv.org/pdf/2306.17775 Algorithm 1
        tds_return_all: whether to return all samples for TDS (default: False)

        guidance_start_step (int): the step to start applying guidance (default: 0)
        guidance_end_step (int): the step to stop applying guidance, -1 means until the end (default: -1)
        
        ### ADDED AFTER REBUTTAL ######
        tds_reweight_temp (float): the temperature for the TDS reweighting (default: 1.0)
        tds_prob_ratio_temp (float): the temperature for the TDS probability ratio (default: 1.0)

        ########################################
    Returns:
        numpy.ndarray: Generated samples of shape [batch_size, D]
    """
    if not mcts:
        assert active_set_size == 1, "active_set_size must be 1 if not using MCTS"
        assert branch_out_size == 1, "branch_out_size must be 1 if not using MCTS"
    
    if tds:
        assert branch_out_size == 1, "branch_out_size should be 1 for TDS"
        # assert predict_on_x1, "predict_on_x1 should be True for TDS"
        assert use_guided_rate, "use_guided_rate should be True for TDS "

    if mask_idx is None:
        mask_idx = S - 1

    B = batch_size
    sample_k = sample_k_for_prob_xt_estimation

    # Sample initial xt
    xt = mask_idx * torch.ones((B, D), dtype=torch.long, device=device)

    t = 0.0
    num_steps = int(1 / dt)  # TODO: Use ceil or int?
    mask_one_hot = torch.zeros((S,), device=device)
    mask_one_hot[mask_idx] = 1.0

    # Treat the case where fixed pads should be used
    pad_mask = None
    if pad_idx is not None:
        pad_one_hot = torch.zeros((S,), device=device)
        pad_one_hot[pad_idx] = 1.0

        # If 'num_unpadded_freq_dict' is not None,
        # sample pads for x0 (=xt at time t=0) and pad xt
        # overwriting the current xt
        if num_unpadded_freq_dict is not None:
            xt, pad_mask = sample_pads_for_x0(xt, pad_idx, num_unpadded_freq_dict)
    # else:
    #     pad_mask = None
    
    xt = xt.unsqueeze(1).repeat(1, active_set_size, 1)
    if pad_mask is not None:
        pad_mask = pad_mask.unsqueeze(1).repeat(1, active_set_size, 1)
    # if predictor_log_prob is not None and PREDICT_ON_X1:
    #     predictor_log_prob = x0_wrapper(predictor_log_prob)

    # log_prob_ratio = None
    gumbel_distribution = torch.distributions.gumbel.Gumbel(torch.tensor(0.0), torch.tensor(1.0))

    all_time_info = defaultdict(list)
    if verbose:
        torch.cuda.synchronize()
        initial_time_info = time.perf_counter()
    valid_steps = 0

    prev_y_prob = None

    ori_mcts, ori_active_set_size, ori_branch_out_size = mcts, active_set_size, branch_out_size 
    ori_tds, ori_svdd, ori_use_guided_rate = tds, svdd, use_guided_rate
    guidance_end_step = num_steps if guidance_end_step == -1 else guidance_end_step
    for step_idx in tqdm(range(num_steps)):
        # Get p(x1 | xt), scaled by temperature
        # This is the unconditional prediction
        # If denoising model trained unconditionally, it doesn't use the cls input
        # If it is trained conditionally, this is the index of the unconditional class

        if step_idx < guidance_start_step or step_idx > guidance_end_step:
            ### reset the parameters for the guidance
            ### no guidance is applied
            mcts = False
            active_set_size = 1
            branch_out_size = 1
            tds = False
            svdd = False
            use_guided_rate = False
            print("RESET: NO GUIDANCE")
        else:
            mcts, active_set_size, branch_out_size = ori_mcts, ori_active_set_size, ori_branch_out_size
            tds, svdd, use_guided_rate = ori_tds, ori_svdd, ori_use_guided_rate
            # print("APPLY: GUIDANCE")

        xt = rearrange(xt, 'B N D -> (B N) D')
        if pad_mask is not None:
            pad_mask = rearrange(pad_mask, 'B N D -> (B N) D')
        B = xt.shape[0]

        if verbose:
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        ##### flop counting
        # flops = FlopCountAnalysis(denoising_model, (xt, t * torch.ones((B,), device=device)))
        # print(flop_count_str(flops))
        # print(flop_count_table(flops))
        ########
        logits = denoising_model(xt, t * torch.ones((B,), device=device))  # (B, D, S)
        if verbose:
            torch.cuda.synchronize()
            all_time_info["denoising_model"].append(time.perf_counter()-start_time)

        pt_x1_probs = F.softmax(logits / x1_temp, dim=-1)  # (B, D, S)

        # If cls free guidance, also get the conditional prediction for the
        # class we want to guide towards
        # p(x1 | xt, y)
        if cond_denoising_model is not None:
            logits_cond = cond_denoising_model(xt, t * torch.ones((B,), device=device))
            pt_x1_probs_cond = F.softmax(logits_cond / x1_temp, dim=-1)

        # Compute the rates and the probabilities
        # See section F.1.1 in DFM

        # When the current state is masked, compute rates for unmasking
        xt_is_mask = (xt == mask_idx).view(B, D, 1).float()
        R_t = (
            xt_is_mask * pt_x1_probs * ((1 + stochasticity * t) / (1 - t))
        )  # (B, D, S)

        # When fixing the pads, do not allow unmasking to padded states by setting
        # the unmasking rates for transitions going to padded states to zero
        if pad_idx is not None:
            R_t *= 1 - pad_one_hot.view(1, 1, -1)

        if cond_denoising_model is not None: ## TODO:
            # Compute conditional rate
            R_t_cond = (
                xt_is_mask * pt_x1_probs_cond * ((1 + stochasticity * t) / (1 - t))
            )  # (B, D, S)

            # Same as in unconditional case above:
            # When fixing the pads, do not allow unmasking to padded states by setting
            # the unmasking rates for transitions going to padded states to zero
            if pad_idx is not None:
                R_t_cond *= 1 - pad_one_hot.view(1, 1, -1)

        if do_purity_sampling:
            # Get purity weight for each dimension for each batch point
            # Only consider dimensions that are currently masked to be eligible for unmasking
            masked_logits = (
                logits * (xt == mask_idx).view(B, D, 1).float()
                + -1e9 * (xt != mask_idx).view(B, D, 1).float()
            )
            # purity_weights: the value of the highest predicted prob at that dimension
            max_masked_logits = torch.max(masked_logits, dim=-1)[0]  # (B, D)
            purity_weights = torch.softmax(
                max_masked_logits / purity_temp, dim=-1
            )  # (B, D)
            # Dimensions with more masks will be upweighted
            R_t *= purity_weights.view(B, D, 1) * torch.sum(
                xt_is_mask, dim=(1, 2)
            ).view(B, 1, 1)
            if cond_denoising_model is not None:
                # Modify conditional rates with purity weights
                masked_logits_cond = (
                    logits_cond * (xt == mask_idx).view(B, D, 1).float()
                    + -1e9 * (xt != mask_idx).view(B, D, 1).float()
                )
                # purity_weights: the value of the highest predicted prob at that dimension
                max_masked_logits_cond = torch.max(masked_logits_cond, dim=-1)[
                    0
                ]  # (B, D)
                purity_weights_cond = torch.softmax(
                    max_masked_logits_cond / purity_temp, dim=-1
                )  # (B, D)
                R_t_cond *= purity_weights_cond.view(B, D, 1) * torch.sum(
                    xt_is_mask, dim=(1, 2)
                ).view(B, 1, 1)

        # When the current state is not a mask, compute rates for remasking
        # Only makes a difference when stochasticity > 0
        remask_rates = (1 - xt_is_mask) * mask_one_hot.view(1, 1, -1) * stochasticity

        # When fixing the pads, do not allow masking of padded states by setting
        # the masking rate for transitions going out of padded states to zero
        if pad_idx is not None:
            xt_is_pad = (xt == pad_idx).view(B, D, 1).float()
            remask_rates *= 1 - xt_is_pad

        R_t += remask_rates

        # Perform predictor guidance by adjusting the unconditional rates
        sub_time_info = defaultdict(list)
        log_prob_ratio = None
        if use_guided_rate and predictor_log_prob is not None:
            ### calculate the guided rates and make sure the predictor is not None
            if predict_on_x1:
                ### whether to use the time-independent predictor (input: x1)
                conditional_R_t, log_prob_ratio, sub_time_info = get_guided_rates_with_predictor_on_x1(
                    predictor_log_prob,
                    xt,
                    t,
                    R_t,
                    S,
                    denoising_model,
                    x1_temp,
                    pad_mask,
                    pad_idx,
                    use_tag=use_tag,
                    guide_temp=guide_temp,
                    verbose=verbose,
                    ########################################
                    #### important parameters
                    sample_k=sample_k,
                    low_threshold=low_threshold,
                    gumbel_softmax_t=gumbel_softmax_t,
                    use_grad_fn_v1=use_grad_fn_v1,
                )
            else: ## use the time-dependent predictor (input: xt, t)
                conditional_R_t, log_prob_ratio, sub_time_info = get_guided_rates(
                    predictor_log_prob,
                    xt,
                    t,
                    R_t,
                    S,
                    use_tag=use_tag,
                    guide_temp=guide_temp,
                    verbose=verbose,
                )
            
            if not sample_xt_with_gumbel_max and not tds:
                R_t = conditional_R_t

            # for key, value in sub_time_info.items():
            #     all_time_info[key].append(value)
            
        
        # Perform predictor-free guidance by using both the unconditional and conditional rates
        if cond_denoising_model is not None: ## may not work for some new cases
            # First add the remask rates to the conditional rates
            # Note the remask rate is the same for unconditional and conditional
            R_t_cond += remask_rates
            # Perform rate adjustment, note that we scale by the inverse temperature
            # If inverse_guide_temp = 0 (guide_temp = inf), equivalent to unconditional
            # If inverse_guide_temp = 1, (guide_temp = 1), equivalent to conditional
            inverse_guide_temp = 1 / guide_temp
            R_t = torch.exp(
                inverse_guide_temp * torch.log(R_t_cond + eps)
                + (1 - inverse_guide_temp) * torch.log(R_t + eps)
            )

        # Set the diagonal of the rates to negative row sum
        R_t.scatter_(-1, xt[:, :, None], 0.0)
        R_t.scatter_(-1, xt[:, :, None], (-R_t.sum(dim=-1, keepdim=True)))

        # Obtain probabilities from the rates
        step_probs = (R_t * dt).clamp(min=0.0, max=1.0)
        step_probs.scatter_(-1, xt[:, :, None], 0.0)
        step_probs.scatter_(  ## TODO: check whether these operations are necessary
            -1,
            xt[:, :, None],
            (1.0 - torch.sum(step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0) ## the sum of step_probs may not be 1.0

        if (tds or sample_xt_with_gumbel_max) and log_prob_ratio is not None:
            conditional_R_t.scatter_(-1, xt[:, :, None], 0.0)
            conditional_R_t.scatter_(-1, xt[:, :, None], (-conditional_R_t.sum(dim=-1, keepdim=True)))

            # conditional_prob_xt_ref = (1.0 + torch.gather(conditional_R_t, -1, xt[:, :, None]) * dt).clamp(min=0.0, max=1.0) ## (B, D, S)
            # conditional_prob_xt_ref = torch.log(conditional_prob_xt_ref+EPS) ## (B, D, S)

            # Obtain probabilities from the rates
            conditional_step_probs = (conditional_R_t * dt).clamp(min=0.0, max=1.0)
            conditional_step_probs.scatter_(-1, xt[:, :, None], 0.0)
            conditional_step_probs.scatter_(  ## TODO: check whether these operations are necessary
                -1,
                xt[:, :, None],
                (1.0 - torch.sum(conditional_step_probs, dim=-1, keepdim=True)).clamp(min=0.0),
            )
            conditional_step_probs = torch.clamp(conditional_step_probs, min=0.0, max=1.0) ## the sum of step_probs may not be 1.0
            
            # if sample_xt_with_gumbel_max:
            conditional_prob_xt = torch.log(torch.gather(conditional_step_probs, -1, xt[:, :, None])+EPS)
            log_prob_ratio.scatter_(-1, xt[:, :, None], 0.0)
            log_prob_ratio.scatter_(-1, xt[:, :, None], conditional_prob_xt - torch.gather(torch.log(step_probs + EPS), -1, xt[:, :, None]))
            # print("difference:", f"{torch.norm(conditional_prob_xt - torch.gather(torch.log(step_probs + EPS), -1, xt[:, :, None]))}")
            # aa = 1
            if tds:
                uncond_step_probs = step_probs.clone()
                step_probs = conditional_step_probs


        # Sample the next xt
        try:
            step_probs = rearrange(step_probs, "(B N) D S -> B N D S", N=active_set_size)
            step_probs = rearrange(repeat(step_probs, 'B N D S -> B N K D S', K=branch_out_size), 'B N K D S -> (B N K) D S')
            if pad_mask is not None:
                pad_mask = rearrange(pad_mask, "(B N) D -> B N D", N=active_set_size)
                pad_mask_K = rearrange(repeat(pad_mask, 'B N D -> B N K D', K=branch_out_size), 'B N K D -> (B N K) D')
            else:
                pad_mask_K = None
                
            if not sample_xt_with_gumbel_max:
                if not sample_max:
                    xt = torch.distributions.Categorical(step_probs).sample()  # (B, D) ## if the sum of step_probs is not 1.0, it will normalize the step_probs
                else: ### argmax sampling for debugging
                    xt = torch.argmax(step_probs, dim=-1)

            else: ## TODO: may need to make sure padded states stay the same
                xt = gumbel_max_sampling(
                    step_probs, gumbel_distribution, 
                    log_prob_ratio=log_prob_ratio, 
                    gamma1=gamma1, gamma2=gamma2,
                    strength_rescale=strength_rescale, 
                    strength_rescale_after_combination=strength_rescale_after_combination,
                    rescale_method=rescale_method,
                    add_gumbel_mean=add_gumbel_mean,
                    gumbel_norm_expectation=gumbel_norm_expectation, 
                    ) ## (B, D)
                if pad_mask_K is not None:
                    ratio_ = torch.sum(xt[pad_mask_K] != pad_idx) / torch.sum(pad_mask_K)
                    if ratio_ > 0:
                        print(f"ratio of invalid pad: {ratio_}")
                        xt[pad_mask_K] = pad_idx

            total_size = active_set_size * branch_out_size
            if mcts and predictor_log_prob is not None: 
                ## only support predict_on_x1=True for now
                assert predict_on_x1, "MCTS only supports predict_on_x1=True for now"
                if t+dt > max_t:
                    sel_num = 1
                    if tds and tds_return_all:
                        sel_num = total_size
                else:
                    sel_num = active_set_size
                # if total_size > sel_num:
                if (not tds and (total_size > sel_num)) or (tds): # and t+dt <= max_t):
                    y_prob, sub_time_info = predictor_based_on_x1(
                        xt, t+dt, predictor_log_prob, denoising_model, x1_temp, 
                        pad_mask_K,  pad_idx, 
                        sample_k=sample_k, threshold=low_threshold,
                        verbose_time=verbose,
                        return_log_prob=False,
                        # **kwargs
                    ) ## p(y|xt) for each candidate
                    y_prob = rearrange(y_prob, "(B NT)-> B NT", NT=active_set_size*branch_out_size)

                    if tds:# and prev_y_prob is not None:
                        y_prob = torch.clamp(y_prob, min=low_threshold)
                        if prev_y_prob is None:
                            prev_y_prob = y_prob
                            sel_indices = torch.arange(total_size)[None, :] #.repeat(batch_size, 1)#.long()
                        else:                    
                            uncond_step_probs = rearrange(repeat(uncond_step_probs, 'BN D S -> BN K D S', K=branch_out_size), 'BN K D S -> (BN K) D S')
                            uncond_trans_prob = torch.gather(uncond_step_probs, -1, xt[:, :, None]).squeeze(-1) ## (B N) D
                            uncond_trans_prob_prod = torch.prod(uncond_trans_prob, dim=-1) ## (B N)
                            uncond_trans_prob_prod = rearrange(uncond_trans_prob_prod, "(B N) -> B N", N=active_set_size*branch_out_size)

                            cond_trans_prob = torch.gather(conditional_step_probs, -1, xt[:, :, None]).squeeze(-1) ## (B N) D
                            cond_trans_prob_prod = torch.prod(cond_trans_prob, dim=-1) 
                            cond_trans_prob_prod = rearrange(cond_trans_prob_prod, "(B N) -> B N", N=active_set_size*branch_out_size)
                            
                            log_weight_ = (torch.log(y_prob) - torch.log(torch.clamp(prev_y_prob, min=low_threshold))) / tds_prob_ratio_temp
                            log_weight_ += torch.log(torch.clamp(uncond_trans_prob_prod, min=low_threshold)) - torch.log(torch.clamp(cond_trans_prob_prod, min=low_threshold))
                            
                            # weight_ = torch.exp(log_weight_)
                            weight_ = torch.softmax(log_weight_ / tds_reweight_temp, dim=-1) ## (B N)  

                            sel_indices = torch.multinomial(weight_, sel_num, replacement=True)
                            prev_y_prob = torch.gather(y_prob, 1, sel_indices) ## (B NK)
                    
                    elif svdd: 
                    # if svdd: ## Categorical sampling
                        # y_prob = torch.clamp(y_prob, min=low_threshold)
                        # y_prob_weight = y_prob ** (1 / svdd_temp)
                        # y_prob_weight = y_prob_weight * torch.min(y_prob_weight, dim=-1, keepdim=True)

                        log_y_prob = torch.log(torch.clamp(y_prob, min=low_threshold))
                        y_prob_weight = torch.softmax(log_y_prob / svdd_temp, dim=-1)
                        # y_prob_weight = torch.clamp(y_prob ** (1 / svdd_temp), min=max((low_threshold ** (1 / svdd_temp), 1e-40)))
                        sel_indices = torch.multinomial(y_prob_weight, sel_num, replacement=False)
                    else:
                        _, sel_indices = torch.topk(y_prob, sel_num, dim=-1)

                
                elif total_size == sel_num: ## no need to select, just use all candidates, best of N
                    sel_indices = torch.arange(total_size)[None, :] #.repeat(batch_size, 1)#.long()

                else:
                    raise ValueError("active_set_size x branch_out_size must be larger than sel_num")
            
            else: ## no MCTS, total_size = active_set_size * branch_out_size = 1
                sel_indices = [0] #0 * torch.ones((B, 1), device=device, dtype=torch.long)
            
            xt = rearrange(xt, "(B NT) D -> B NT D", NT=total_size)
            # xt = rearrange(xt, "(B NT) D -> B NT D", NT=active_set_size*branch_out_size)
            B = xt.shape[0]
            xt = xt[torch.arange(B)[:, None], sel_indices] # (B, N, D)
    
            # xt = torch.gather(xt, dim=1, index=sel_indices.unsqueeze(-1).repeat(1,1,xt.shape[-1]))
        except ValueError:
            raise ValueError(
                "ValueError in 'torch.distributions.Categorical(step_probs).sample()', step_probs might not valid."
            )
        for key, value in sub_time_info.items():
            all_time_info[key].append(value)

        # print(f"xt: {xt.unique()}")
        t += dt
        valid_steps += 1
        if t > max_t:
            # xt = xt[:, 0, :] # (B, D)
            # xt = xt.reshape(-1, xt.shape[-1])
            xt = torch.flatten(xt, start_dim=0, end_dim=1)  # (B, N*D)
            break

    if verbose and len(all_time_info) > 0:
        torch.cuda.synchronize()
        total_time = time.perf_counter() - initial_time_info
        # assert step_idx == num_steps - 1, "step_idx should be equal to num_steps - 1"
        all_time_info["batch_inference_time_per_step"] = total_time / valid_steps
        all_time_info["sample_inference_time_per_step"] = total_time / (batch_size * valid_steps)

        denoising_model_time = all_time_info.pop("denoising_model")
        all_time_info["batch_denoising_model_time"] = np.sum(denoising_model_time)
        all_time_info["batch_denoising_model_time_per_step"] = np.mean(denoising_model_time)
        all_time_info["sample_denoising_model_time_per_step"] = np.mean(denoising_model_time) / batch_size ## TODO: debug for MCTS
        all_time_info["batch_size"] = batch_size
        all_time_info["valid_steps"] = valid_steps

        if len(sub_time_info) > 0:
            for key, value in sub_time_info.items():
                # all_time_info["mean " + key] = np.mean(value)
                all_time_info["mean " + key] = np.mean(value)
                all_time_info.pop(key)

    # For any state that is still masked, take the argmax of the logits
    # of the final xt
    if argmax_final: ## TODO: check how to fit MCTS
        B = xt.shape[0]
        xt_is_mask = (xt == mask_idx).view(B, D).float()
        logits = denoising_model(xt, t * torch.ones((B,), device=device))  # (B, D, S)
        if cond_denoising_model is not None:
            logits_cond = cond_denoising_model(xt, t * torch.ones((B,), device=device))
            logits = (
                inverse_guide_temp * logits_cond + (1 - inverse_guide_temp) * logits
            )
        xt = torch.argmax(logits, dim=-1) * xt_is_mask + xt * (1 - xt_is_mask)

    return xt.detach().cpu().numpy(), all_time_info

########################################################################################
### 1. Gumbel-max based sampling + rescale
def gumbel_max_sampling(
        step_probs, gumbel_distribution, 
        log_prob_ratio=None, 
        gamma1=1, gamma2=1, 
        strength_rescale=False, 
        strength_rescale_after_combination=False, 
        rescale_method="new", 
        add_gumbel_mean=False,
        gumbel_norm_expectation=None, 
        # random_g_cutoff=80.0
        ):
    
    '''
        Gumbel-max based sampling + rescale
        step_probs: probability distribution for sampling next states (B, D, S)
        gumbel_distribution: gumbel distribution for sampling gumbel noise (0, 1)
        log_prob_ratio: log probability ratio logp(y|x_t-1) - logp(y|x_t) (B, D, S)
        gamma1: the weight for gumbel noise
        gamma2: the weight for log probability ratio
        strength_rescale: whether to rescale the strength of log probability ratio
        rescale_method: the method for rescaling the strength of the gumbel noise-like features (default: "new")
        add_gumbel_mean: whether to add the mean of the gumbel noise to the rescaled gumbel noise-like features (should not matter)
        gumbel_norm_expectation: the expectation of the gumbel noise (only used for old rescale method)

    '''
    # log_probs = torch.log(step_probs + EPS)

    if gamma1 != 0:
        gumbel_sampled = gumbel_distribution.sample(step_probs.shape).to(step_probs.device)
    else:
        gumbel_sampled = torch.zeros_like(step_probs, device=step_probs.device)
    

    if gamma2 != 0 and log_prob_ratio is not None:

        if strength_rescale:
            # assert gumbel_norm_expectation is not None, "gumbel_norm_expectation must be provided if strength_rescale is True"
            log_prob_ratio = rescale_strength_method(
                log_prob_ratio, gumbel_norm_expectation, 
                resale_method=rescale_method, add_gumbel_mean=add_gumbel_mean
                )
            
    else:
        gamma2 = 0
        log_prob_ratio = torch.zeros_like(step_probs, device=step_probs.device)
    
    random_g = gamma1 * gumbel_sampled + gamma2 * log_prob_ratio #- gumbel_sampled) ## (B, D, S)
    ## [For Debugging] TODO: CHECK.  Avoid overflow of the exponential 
    # random_g = torch.clamp(random_g, max=random_g_cutoff)
    if strength_rescale_after_combination:
        random_g = rescale_strength_method(
            random_g, gumbel_norm_expectation, 
            resale_method=rescale_method, add_gumbel_mean=add_gumbel_mean
            )
        # random_g = random_g / (torch.norm(random_g, dim=-1, keepdim=True) + EPS) * gumbel_norm_expectation

    # sampled_z = (log_probs + random_g).argmax(-1) ## (B, D)
    sampled_z = (torch.exp(random_g) * step_probs).argmax(-1)
    if torch.sum(torch.gather(step_probs, -1, sampled_z[:,:,None]) == 0) > 0:
        print("Some unlikely states are sampled.")
    return sampled_z

def rescale_strength_method(g, gumbel_norm_expectation=None, resale_method="new", add_gumbel_mean=False):
    ### rescale the strength of the gumbel noise-like features ###
    if resale_method == "old":
        assert gumbel_norm_expectation is not None, "gumbel_norm_expectation must be provided if strength_rescale is True"
        gumbel_norm_expectation = gumbel_norm_expectation.to(device=g.device)
        g_rescaled = g / (torch.norm(g, dim=-1, keepdim=True) + EPS) * gumbel_norm_expectation ## (B, D, S)
    elif resale_method == "new":
        g_rescaled = (g - torch.mean(g, dim=-1, keepdim=True)) / (torch.std(g, dim=-1, keepdim=True)+ EPS) * GUMBEL_STD.to(device=g.device) + (GUMBEL_MEAN.to(device=g.device) if add_gumbel_mean else 0)
    return g_rescaled

########################################################################################
# 2. Get guided rates with time-dependent predictor p(y|xt,t)

def get_guided_rates(
    predictor_log_prob: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xt: torch.Tensor,  # Shape: (B, D)
    t: float,
    R_t: torch.Tensor,  # Shape: (B, D, S)
    S: int,
    use_tag: bool = False,
    guide_temp: float = 1.0,
    log_prob_ratio_cutoff: float = 80.0,
    verbose = False,
) -> torch.Tensor:
    """
    Computes guide-adjusted rates with time-dependent predictor p(y|xt,t) for predictor guidance.

    Implements both exact guidance by computing likelihood ratios for all possible transitions,
    and Taylor-approximated guidance (TAG) using gradients of the predictor.

    Args:
        predictor_log_prob (callable): Function that takes (x,t) and returns log p(y|x,t)
        xt (torch.Tensor): Current states of shape (B, D)
        t (float): Current time
        R_t (torch.Tensor): Unconditional rates of shape (B, D, S)
        S (int): Size of categorical state space
        use_tag (bool, optional): Whether to use Taylor approximation. Defaults to False.
        guide_temp (float, optional): Guidance temperature. Defaults to 1.
        log_prob_ratio_cutoff (float, optional): Maximum value for log ratios. Defaults to 80.
        verbose (bool, optional): Whether to print debug info. Defaults to False.

    Returns:
        torch.Tensor: Guide-adjusted rates of shape (B, D, S)
    """
    B, D = xt.shape
    device = xt.device
    t = t * torch.ones((B,), device=device)
    time_info = {}
    if not use_tag:
        # Exact guidance case
        # log p(y|x=z_t), shape (B,)
        if verbose:
            torch.cuda.synchronize()
            start_time = time.perf_counter()    
        log_prob_xt = predictor_log_prob(xt, t)

        if verbose:
            torch.cuda.synchronize()
            pred_log_prob_xt_t = time.perf_counter()-start_time
            time_info["[xt] pred_log_prob_xt_batch_time"] = pred_log_prob_xt_t
            time_info["[xt] pred_log_prob_xt_per_sample_time"] = pred_log_prob_xt_t / B
            start_time = time.perf_counter()
        # Get all jump transitions, shape (B*D*S, D)
        xt_jumps = get_all_jump_transitions(xt, S)

        if verbose:
            torch.cuda.synchronize()
            get_jump_trans_t = time.perf_counter()-start_time
            time_info["get_jump_trans_batch_time"] = get_jump_trans_t
            start_time = time.perf_counter()

        # Get log probs for all transitions
        # Shape: (B*D*S,) -> (B, D, S)

        log_prob_xt_jumps = predictor_log_prob(
            xt_jumps, t.repeat(1, D * S).flatten()
        ).view(B, D, S) ## TODO: check whether add mask

        if verbose:
            torch.cuda.synchronize()
            pred_log_prob_xt_jumps_t = time.perf_counter()-start_time
            time_info["[xt jumps] pred_log_prob_xt_jumps_batch_time"] = pred_log_prob_xt_jumps_t

        # Compute log ratios
        # Shape (B, D, S)
        log_prob_ratio = log_prob_xt_jumps - log_prob_xt.view(B, 1, 1)

    else:
        # Taylor-approximated guidance (TAG) case
        # One-hot encode categorical data, shape (B, D, S)
        xt_ohe = F.one_hot(xt.long(), num_classes=S).to(torch.float)

        # \grad_{x}{log p(y|x)}(z_t), shape (B, D, S)
        with torch.enable_grad():
            xt_ohe.requires_grad_(True)
            # log p(y|x=z_t), shape (B,)
            if verbose:
                torch.cuda.synchronize()
                start_time = time.perf_counter()   
            log_prob_xt_ohe = predictor_log_prob(xt_ohe, t)
            if verbose:
                torch.cuda.synchronize()
                pred_log_prob_xt_t = time.perf_counter()-start_time
                time_info["[xt] pred_log_prob_xt_batch_time"] = pred_log_prob_xt_t            
                time_info["[xt] pred_log_prob_xt_per_sample_time"] = pred_log_prob_xt_t / B

            log_prob_xt_ohe.sum().backward()
            # Shape (B, D, S)
            grad_log_prob_xt_ohe = xt_ohe.grad
        # 1st order Taylor approximation of the log difference
        # Shape (B, D, S)
        log_prob_ratio = grad_log_prob_xt_ohe - (xt_ohe * grad_log_prob_xt_ohe).sum(
            dim=-1, keepdim=True
        )

    # Scale log prob ratio by temperature
    log_prob_ratio /= guide_temp

    # Clamp the log prob ratio to avoid overflow in exp
    log_prob_ratio = torch.clamp(log_prob_ratio, max=log_prob_ratio_cutoff)
    # Exponentiate to get p(y|x=z~) / p(y|x=z_t)
    prob_ratio = torch.exp(log_prob_ratio)
    # Multiply the reverse rate elementwise with the density ratio
    # Note this doesn't deal with the diagonals
    R_t = R_t * prob_ratio

    if R_t.isnan().any():
        raise ValueError(f"The rate matrix 'R_t' contains NaNs.")

    return R_t, log_prob_ratio, time_info

########################################################################################
# 3. Get guided rates with time-independent predictor p(y|x1)

def get_guided_rates_with_predictor_on_x1(
    predictor_log_prob: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xt: torch.Tensor,  # Shape: (B, D)
    t: float,
    R_t: torch.Tensor,  # Shape: (B, D, S)
    S: int,
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x1_temp: float,
    pad_mask: torch.Tensor,
    pad_idx: int,
    use_tag: bool = False,
    guide_temp: float = 1.0,
    log_prob_ratio_cutoff: float = 80.0,
    ###### important arguments ######
    sample_k: int = 10,
    low_threshold: float = 1e-8,
    gumbel_softmax_t: float = 1.0,
    use_grad_fn_v1: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Computes guide-adjusted rates with time-independent/training-free (x1) predictor.

    Implements both exact guidance by 

        computing likelihood ratios for all possible transitions (method 1),
    and 
        Taylor-approximated guidance (TAG) using gradients of the predictor (method 2).

    Args:
        predictor_log_prob (callable): Function that takes x1 and returns log p(y|x1)
        xt (torch.Tensor): Current states of shape (B, D)
        t (float): Current time
        R_t (torch.Tensor): Unconditional rates of shape (B, D, S)
        S (int): Size of categorical state space
        use_tag (bool, optional): Whether to use Taylor approximation. Defaults to False.
        guide_temp (float, optional): Guidance temperature. Defaults to 1.
        log_prob_ratio_cutoff (float, optional): Maximum value for log ratios. Defaults to 80.
        
        sample_k (int, optional): the number of sampled x1 to estimate the log p(y|xt) through Monte Carlo sampling. Defaults to 10.
        threshold: minimum contraints for the p(y|xt) to avoid log(0).
        verbose (bool, optional): Whether to print info about execution time. Defaults to False.
        use_grad_fn_v1 (bool, optional): whether to use the first version of gradient-based guidance (default as False)
        gumbel_softmax_t: temperature for gumbel softmax to estimate the sampling process


    Returns:
        torch.Tensor: Guide-adjusted rates of shape (B, D, S)
        torch.Tensor: log of the probability ratio of shape (B, D)
        dict: time information  
    """
    B, D = xt.shape
    device = xt.device
    t = t * torch.ones((B,), device=device)
    time_info = {}
    if not use_tag:
        log_prob_xt, time_info_xt = predictor_based_on_x1(
            xt, t, predictor_log_prob, denoising_model, x1_temp, 
            pad_mask,  pad_idx,
            sample_k=sample_k, threshold=low_threshold,
            verbose_time=verbose
            )
        if verbose:
            if len(time_info_xt) > 0:
                time_info.update({"[xt] " + key: time_info_xt[key] for key in time_info_xt})
            torch.cuda.synchronize()
            start_time = time.perf_counter()
        xt_jumps, pad_mask_jumps = get_all_jump_transitions(xt, S, pad_mask=pad_mask) 

        if verbose:
            torch.cuda.synchronize()
            get_jump_trans_t = time.perf_counter()-start_time
            time_info["get_jump_trans_batch_time"] = get_jump_trans_t

        ## TODO: remove the vanilla xt from thhe xt_jumps
        ## OR merge xt_jumps with xt

        
        log_prob_xt_jumps, time_info_xt_jump = predictor_based_on_x1(
            xt_jumps, t.repeat(1, D * S).flatten(), predictor_log_prob,
            denoising_model, x1_temp, pad_mask_jumps, pad_idx,
            sample_k=sample_k, threshold=low_threshold, 
            verbose_time=verbose
        )
        log_prob_xt_jumps = log_prob_xt_jumps.view(B, D, S)
        if verbose:
            if len(time_info_xt_jump) > 0:
                time_info.update({"[xt jumps] " + key: time_info_xt_jump[key] for key in time_info_xt_jump})
        # Compute log ratios
        # Shape (B, D, S)
        log_prob_ratio = log_prob_xt_jumps - log_prob_xt.view(B, 1, 1)

    else:
        # Taylor-approximated guidance (TAG) case
        # One-hot encode categorical data, shape (B, D, S)
        xt_ohe = F.one_hot(xt.long(), num_classes=S).to(torch.float)

        grad_fn = grad_fn_v1 if use_grad_fn_v1 else grad_fn_v2
        grad_log_prob_xt_ohe, time_info_xt = grad_fn(
            xt_ohe, t, predictor_log_prob, x1_temp, denoising_model, S, 
            pad_mask=pad_mask, pad_idx=pad_idx, sample_k=sample_k, threshold=low_threshold,
            gumbel_softmax_t=gumbel_softmax_t, verbose_time=verbose
            )
        if len(time_info_xt) > 0:
            time_info.update({"[xt] " + key: time_info_xt[key] for key in time_info_xt})

        log_prob_ratio = grad_log_prob_xt_ohe - (xt_ohe * grad_log_prob_xt_ohe).sum(
            dim=-1, keepdim=True
        )

    # Scale log prob ratio by temperature
    log_prob_ratio /= guide_temp

    # Clamp the log prob ratio to avoid overflow in exp  ## TODO: decide the cutoff value
    log_prob_ratio = torch.clamp(log_prob_ratio, max=log_prob_ratio_cutoff)

    # Exponentiate to get p(y|x=z~) / p(y|x=z_t)
    prob_ratio = torch.exp(log_prob_ratio)
    # print(f"prob_ratio: max {prob_ratio.view(B, -1).max(-1)[0]} | min {prob_ratio.view(B, -1).min(-1)[0]}")

    # Multiply the reverse rate elementwise with the density ratio
    # Note this doesn't deal with the diagonals
    R_t = R_t * prob_ratio
    
    if torch.isnan(R_t).any():
        raise ValueError(f"The rate matrix 'R_t' contains NaNs.")
        
    # return R_t, time_info
    return R_t, log_prob_ratio, time_info


def predictor_based_on_x1(
        xt, t, predictor_log_prob, denoising_model, x1_temp, 
        pad_mask=None, pad_idx=None, sample_k=10, threshold=1e-9, verbose_time=False, return_log_prob=True, 
    ):
    """
    Using Time-independent Predictor p(y|x1) to estimate p(y|xt) for training-free guidance.

    Args:
        xt (torch.Tensor): Current states of shape (B, D) at timestep t
        t (float): Current time
        predictor_log_prob: time-independent predictor function that takes x1 and returns log p(y|x1)
        denoising_model (torch.nn.Module): flow model
        x1_temp (float, optional): Temperature for x1 prediction logits. Defaults to 1.0.
        pad_mask (torch.Tensor, optional): Mask for padded states. Defaults to None.
        pad_idx (int, optional): Index of padded states. Defaults to None.
        sample_k (int, optional): the number of sampled x1 to estimate the log p(y|xt) through Monte Carlo sampling. Defaults to 10.
        threshold: minimum contraints for the p(y|xt) to avoid log(0).
        verbose_time (bool, optional): Whether to print info about execution time. Defaults to False.
        return_log_prob (bool, optional): Whether to return log probabilities or probabilities. Defaults to True.

    Returns:
        torch.Tensor: p(y|xt) or log p(y|xt) of shape (B,)
        dict: time info (it is empty if verbose_time is False)

    """
    
    B = xt.shape[0]
    time_info = {}
    if verbose_time:
        torch.cuda.synchronize()
        start_time_ori = start_time = time.perf_counter()
    logits = denoising_model(xt, t * torch.ones((B,), device=xt.device))
    
    if verbose_time:
        torch.cuda.synchronize()
        denoise_model_t = time.perf_counter()-start_time
        time_info["estimating log(p(y|xt)): denoise_model_batch_time"] = denoise_model_t
        time_info["estimating log(p(y|xt)): denoise_model_per_sample_time"] = denoise_model_t / B
        # print(f"Time for denoising_model: {denoise_model_t}")
        start_time = time.perf_counter()

    pt_x1_probs = F.softmax(logits / x1_temp, dim=-1)  # (B, D, S)

    # x1_k = torch.distributions.OneHotCategorical(probs=repeat(pt_x1_probs, 'bs n atomnames -> bs k n atomnames', k=k)).sample()
    # x1_k = torch.argmax(x1_k, dim=-1) ## bs k n

    # x1_k = torch.distributions.Categorical(probs=repeat(pt_x1_probs, 'bs n atomnames -> bs k n atomnames', k=k)).sample()
    # x1_k = torch.distributions.Categorical(probs=pt_x1_probs).sample((k,)).transpose(0,1)

    # x1_k = fast_categorical(repeat(pt_x1_probs, 'bs n atomnames -> bs k n atomnames', k=k))
    x1_k = fast_categorical(p=pt_x1_probs, k=sample_k, squeeze=False).permute(0, 2, 1)

    if pad_mask is not None and pad_idx is not None:
        x1_k[repeat(pad_mask, 'bs n -> bs k n', k=sample_k)] = pad_idx

    x1_k = rearrange(x1_k, 'bs k n -> (bs k) n')


    if verbose_time:
        torch.cuda.synchronize()
        sampling_t = time.perf_counter()-start_time
        time_info["estimating log(p(y|xt)): sampling_x1_k_batch_time"] = sampling_t
        start_time = time.perf_counter()
    log_prob_x1_k = predictor_log_prob(x1_k)
    if verbose_time:
        torch.cuda.synchronize()
        predictor_log_prob_t = time.perf_counter()-start_time
        time_info["estimating log(p(y|xt)): predictor_log_prob_x1_k_batch_time"] = predictor_log_prob_t
        time_info["estimating log(p(y|xt)): predictor_log_prob_x1_per_sample_time"] = predictor_log_prob_t / x1_k.shape[0]

    log_prob_x1_k = rearrange(log_prob_x1_k, '(bs k) -> bs k', bs=B)


    prob_x1_k = torch.exp(log_prob_x1_k) #.to(torch.float64)) ## TODO: check whether to convert to float64
    # prob_xt = torch.mean(prob_x1_k, dim=-1) #prob_x1_k.mean(dim=-1)
    prob_xt = torch.sum(prob_x1_k, dim=-1) / (torch.sum(prob_x1_k != 0, dim=-1) + 1e-9) ## TODO: check which is the better way to compute the mean
    assert not torch.isnan(prob_xt).any()
    # log_prob_xt = torch.log(prob_xt+threshold)
    log_prob_xt = torch.log(torch.clamp(prob_xt, min=threshold))

    if verbose_time:
        torch.cuda.synchronize()
        predictor_log_prob_xt_t = time.perf_counter()-start_time_ori
        time_info["estimating log(p(y|xt)): total batch time"] = predictor_log_prob_xt_t
        time_info["estimating log(p(y|xt)): total time per sample"] = predictor_log_prob_xt_t / B
    if not return_log_prob:
        return prob_xt, time_info
    
    return log_prob_xt, time_info


def grad_fn_v2(
        xt_ohe, t, predictor_log_prob, x1_temp, denoising_model, S, 
        pad_mask=None, pad_idx=None, sample_k=10, threshold=1e-9, gumbel_softmax_t=1.0,
        verbose_time=False
    ):
   
    """
    Get the gradient of log p(y|xt) w.r.t. xt for training-free guidance.

    Implements both exact guidance by computing likelihood ratios for all possible transitions,
    and Taylor-approximated guidance (TAG) using gradients of the predictor.

    Args:
        xt_ohe (torch.Tensor): Current states of shape (B, D) at timestep t
        t (float): Current time
        predictor_log_prob: time-independent predictor function that takes x1 and returns log p(y|x1)
        x1_temp (float, optional): Temperature for x1 prediction logits. Defaults to 1.0.
        denoising_model (torch.nn.Module): flow model
        S: number of classes
        pad_mask (torch.Tensor, optional): Mask for padded states. Defaults to None.
        pad_idx (int, optional): Index of padded states. Defaults to None.
        sample_k (int, optional): the number of sampled x1 to estimate the log p(y|xt) through Monte Carlo sampling. Defaults to 10.
        threshold: minimum contraints for the p(y|xt) to avoid log(0).
        gumbel_softmax_t: temperature for gumbel softmax

    Returns:
        torch.Tensor: p(y|xt) or log p(y|xt) of shape (B,)
        dict: time info (it is empty if verbose_time is False)

    """

    B = xt_ohe.shape[0]
    xt_ohe = xt_ohe.clone().detach()
    time_info = {}
    with torch.enable_grad():
        xt_ohe.requires_grad_(True)

        if verbose_time:
            torch.cuda.synchronize()
            start_time_ori = start_time = time.perf_counter()
        logits = denoising_model(xt_ohe, t * torch.ones((B,), device=xt_ohe.device)) #, is_x_onehot=True)
        
        if verbose_time:
            torch.cuda.synchronize()
            denoise_model_t = time.perf_counter()-start_time
            time_info["estimating log(p(y|xt)): denoise_model_batch_time"] = denoise_model_t
            time_info["estimating log(p(y|xt)): denoise_model_per_sample_time"] = denoise_model_t / B
            # print(f"Time for denoising_model: {denoise_model_t}")
            start_time = time.perf_counter()

        logits = logits / x1_temp

        # repeat_shape = list(logits.shape)
        # repeat_shape.insert(1, k)
        # gumbel_sample = sample_gumbel(repeat_shape).to(logits.device)

        logits = logits.unsqueeze(1)

        x1_k_ohe = torch.nn.functional.gumbel_softmax(logits.repeat((1, sample_k, 1, 1)), hard=True, tau=gumbel_softmax_t, dim=-1)
        x1_k_ohe = rearrange(x1_k_ohe, 'bs k n c -> (bs k) n c')

        if pad_mask is not None and pad_idx is not None:
            pad_mask_k = repeat(pad_mask, 'bs n -> (bs k) n', k=sample_k)
            pad_idx_k = torch.nn.functional.one_hot(pad_idx * torch.ones_like(pad_mask_k), num_classes=S).float()
            pad_mask_k = repeat(pad_mask_k, 'bsk n -> bsk n c', c=S).float()
            x1_k_ohe_m = x1_k_ohe * (1 - pad_mask_k) + pad_idx_k * pad_mask_k
        else:
            x1_k_ohe_m = x1_k_ohe

        if verbose_time:
            torch.cuda.synchronize()
            sampling_t = time.perf_counter()-start_time
            time_info["estimating log(p(y|xt)): sampling_x1_k_batch_time"] = sampling_t
            start_time = time.perf_counter()

        # x1_k_ohe_m = rearrange(x1_k_ohe_m, 'bs k n c -> (bs k) n c')
        ############ # Not working for the molecule cases
        # flops = FlopCountAnalysis(predictor_log_prob, x1_k_ohe_m)
        # print(flop_count_str(flops))
        # print(flop_count_table(flops))
        ############
        log_prob_x1_k_ohe = predictor_log_prob(x1_k_ohe_m)
        if verbose_time:
            torch.cuda.synchronize()
            predictor_log_prob_t = time.perf_counter()-start_time
            time_info["estimating log(p(y|xt)): predictor_log_prob_x1_k_batch_time"] = predictor_log_prob_t
            time_info["estimating log(p(y|xt)): predictor_log_prob_x1_per_sample_time"] = predictor_log_prob_t / x1_k_ohe_m.shape[0]

        log_prob_x1_k_ohe = rearrange(log_prob_x1_k_ohe.view(-1), '(bs k) -> bs k', bs=B)  ## [Fix the bug]
        prob_x1_k_ohe = torch.exp(log_prob_x1_k_ohe) #.to(torch.float64)) ## TODO: check whether to convert to float64
        prob_xt_ohe = prob_x1_k_ohe.mean(dim=-1)

        # log_prob_xt_ohe = torch.log(prob_xt_ohe+threshold)
        log_prob_xt_ohe = torch.log(torch.clamp(prob_xt_ohe, min=threshold))
        if verbose_time:
            torch.cuda.synchronize()
            predictor_log_prob_xt_t = time.perf_counter()-start_time_ori
            time_info["estimating log(p(y|xt)): total batch time"] = predictor_log_prob_xt_t
            time_info["estimating log(p(y|xt)): total time per sample"] = predictor_log_prob_xt_t / B
            start_time = time.perf_counter()

        xt_grad = torch.autograd.grad(log_prob_xt_ohe.sum(), xt_ohe)[0]
        if verbose_time:
            torch.cuda.synchronize()
            grad_t = time.perf_counter()-start_time
            time_info["backprop log(p(y|xt)): grad_batch_time"] = grad_t
            time_info["backprop log(p(y|xt)): grad_per_sample_time"] = grad_t / B
        try:
            assert not torch.isnan(xt_grad).any()
        except Exception as e:
            print(f"Xt Grad Error: {e}")

        # ## TODO: rescale_grad
        # # xt_grad = rescale_grad(xt_grad, mask, clip_scale=1.0) * self.rho ## Maybe useful

        return xt_grad, time_info

def fast_categorical(p, k=1, squeeze=True):
    '''
    Fast sampling from a categorical distribution

    Reference:
        https://github.com/pytorch/pytorch/issues/30968#issuecomment-2312590267
    '''
    rand_vals = torch.rand(list(p.shape[:-1]) + [k], device=p.device)
    pdf = p.cumsum(dim=-1)
    pdf /= pdf[..., -1:] ## To make sure the last element is 1.0
    samples = torch.searchsorted(pdf, rand_vals, right=False) ## batch_size, seq_len, k
    if samples.shape[-1] == 1 and squeeze:
        samples = samples.squeeze(-1)
    # try:
    #     assert torch.max(samples) < p.shape[-1] and torch.min(samples) >= 0
    # except Exception as e:
    #     samples = torch.clamp(samples, min=0, max=p.shape[-1]-1)
    return samples

def get_all_jump_transitions(
    xt: torch.Tensor,  # Shape: (B, D)
    S: int,
    pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:  # Shape: (B*D*S, D)
    """
    Gets all possible single-dimension transitions from current states.

    Creates a tensor containing all possible states that differ from input states
    in exactly one position, for each sequence in the batch.

    Args:
        xt: Current state tensor of shape (batch_size, sequence_length)
        S: Size of categorical state space (number of possible values per position)

    Returns:
        Tensor of shape (batch_size * sequence_length * state_space, sequence_length)
        containing all possible single-token transitions
    """
    B, D = xt.shape
    device = xt.device

    # Create B*D*S copies of input sequences
    # Shape: (B, 1, D) -> (B, D*S, D)
    xt_expand = xt.unsqueeze(1).repeat(1, D * S, 1)
    if pad_mask is not None:
        pad_mask = pad_mask.unsqueeze(1).repeat(1, D * S, 1)
        pad_mask = pad_mask.view(-1, D)
    # Flatten batch and transition dimensions
    # Shape: (B, D*S, D) -> (B*D*S, D)
    xt_expand = xt_expand.view(-1, D)

    # Create indices for all possible transitions
    # Shape: (D*S,) -> (B, D*S) -> (B*D*S,)
    jump_idx = torch.arange(D * S).to(device)
    jump_idx = jump_idx.repeat(B, 1).flatten()

    # Create tensor for states after one transition
    xt_jumps = xt_expand.clone()

    # Calculate which dimension changes for each transition
    # Shape: (B*D*S,)
    jump_dims = jump_idx // S

    # Calculate new value for changed dimension
    # Shape: (B*D*S,)
    jump_states = jump_idx % S

    # Apply transitions by assigning new values at transition dimensions
    # Shape: (B*D*S, D)
    xt_jumps[
        torch.arange(jump_idx.size(0), device=device),
        jump_dims,  # Index the transitioned dimension
    ] = jump_states  # Assign the new state

    return xt_jumps, pad_mask
    # if pad_mask is not None:
    #     return xt_jumps, pad_mask
    # return xt_jumps, None


def flow_matching_loss_masking(
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x1: torch.Tensor,
    mask_idx: int,
    reduction: str = "mean",
    pad_idx: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Computes flow matching cross entropy loss for masked denoising.

    Args:
        denoising_model: Model that takes (x,t) and returns logits for each position
        x1: Target sequence tensor of shape (B, D)
        mask_idx: Index representing the mask token
        reduction: Reduction for cross entropy loss - 'none', 'mean', or 'sum'
        pad_idx: Optional index representing padding token, which will be preserved during noising
        loss_mask: Optional boolean mask of shape (B, D) indicating which positions to include in loss
        label_smoothing: Label smoothing parameter for cross entropy loss

    Returns:
        Loss tensor (scalar if reduction is 'mean'/'sum', shape (B,) if 'none')
    """
    B, D = x1.shape
    # Sample random timestep t \in [0,1] for each sequence in batch
    t = torch.rand((B,)).to(x1.device)

    # Sample xt by masking x1 according to t
    xt = sample_xt(x1, t, mask_idx, pad_idx)

    # Get model predictions
    logits = denoising_model(xt, t)  # (B, D, S)

    # Create mask for positions to exclude from loss:
    # - Positions that are not masked in xt (already revealed)
    # - Positions marked False in loss_mask (if provided)
    exclude = xt != mask_idx
    if loss_mask is not None:
        exclude = torch.logical_or(exclude, ~loss_mask)

    # Copy x1 to avoid modifying the input
    x1 = x1.clone()
    # Set excluded positions to -1 so they're ignored by cross entropy loss
    x1[exclude] = -1

    loss = F.cross_entropy(
        logits.transpose(1, 2),
        x1,
        reduction=reduction,
        ignore_index=-1,
        label_smoothing=label_smoothing,
    )
    return loss


def eval_nll(
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x1: torch.Tensor,
    mask_idx: int,
    timesteps: int = 100,
    # reduction: str = "mean",
    pad_idx: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Computes flow matching cross entropy loss for masked denoising.

    Args:
        denoising_model: Model that takes (x,t) and returns logits for each position
        x1: Target sequence tensor of shape (B, D)
        mask_idx: Index representing the mask token
        pad_idx: Optional index representing padding token, which will be preserved during noising
        loss_mask: Optional boolean mask of shape (B, D) indicating which positions to include in loss
        label_smoothing: Label smoothing parameter for cross entropy loss

    Returns:
        Loss tensor (scalar if reduction is 'mean'/'sum', shape (B,) if 'none')
    """
    B, D = x1.shape
    # Sample random timestep t \in [0,1] for each sequence in batch

    nll = 0.0
    from tqdm import trange

    # valid_token_num = 0
    token_nll_list = []
    seq_nll_array = np.zeros((B,), dtype=np.float32)
    # seq_nll_array_n = np.zeros((B,), dtype=np.float32)
    # loss_cal_token_num = np.zeros((B,), dtype=np.int32)
    # mean_nll = 0.0
    for i in range(timesteps-1):
        t = 1.0 * (i+1) / timesteps
        t = t * torch.ones((B,)).to(x1.device)

        with torch.no_grad():
            # t = torch.rand((B,)).to(x1.device)

            # Sample xt by masking x1 according to t
            xt = sample_xt(x1, t, mask_idx, pad_idx)

            # Get model predictions
            logits = denoising_model(xt, t)  # (B, D, S)

            # Create mask for positions to exclude from loss:
            # - Positions that are not masked in xt (already revealed)
            # - Positions marked False in loss_mask (if provided)
            exclude = xt != mask_idx
            if loss_mask is not None:
                exclude = torch.logical_or(exclude, ~loss_mask)

            # Copy x1 to avoid modifying the input
            x1_copy = x1.detach().clone()
            # Set excluded positions to -1 so they're ignored by cross entropy loss
            x1_copy[exclude] = -1

            loss = F.cross_entropy(
                logits.transpose(1, 2),
                # x1,
                x1_copy.long(),  # Ensure x1 is long for cross entropy
                reduction="none", #"none",
                ignore_index=-1,
                label_smoothing=label_smoothing,
            )
        seq_nll_array += (loss.sum(dim=1)  / ( (x1_copy != -1).sum(dim=1) + 1e-9)).cpu().numpy()
        # loss_cal_token_num += (x1_copy != -1).sum(dim=1).cpu().numpy()
        # seq_nll_array_n += (loss.sum(dim=1)).cpu().numpy()
        token_nll_list.extend(loss[~exclude].cpu().tolist())

    seq_nll_array /= (timesteps - 1)
    # seq_nll_array_n /= loss_cal_token_num + 1e-9

    print(f"Eval NLL: mean per-token NLL over all tokens ={np.mean(token_nll_list)} (std: {np.std(token_nll_list)}) , total tokens={len(token_nll_list)}")
    print(f"Eval NLL: mean per-token NLL over all sequences={np.mean(seq_nll_array)} (std: {np.std(seq_nll_array)}) , total sequences={len(seq_nll_array)}")
    # print(f"Eval NLL: mean per-token NLL over all sequences (weighted)={np.mean(seq_nll_array_n)} (std: {np.std(seq_nll_array_n)}) , total sequences={len(seq_nll_array_n)}")
    return token_nll_list, seq_nll_array

def sample_xt(
    x1: torch.Tensor, t: torch.Tensor, mask_idx: int, pad_idx: Optional[int] = None
) -> torch.Tensor:
    """
    Samples a noised state xt by masking x1 according to time t.

    Args:
        x1: Input sequence tensor of shape (B, D)
        t: Time values tensor of shape (B,)
        mask_idx: Index representing the mask token
        pad_idx: Optional index representing padding token that should be preserved

    Returns:
        Noised sequence tensor of shape (B, D)
    """
    B, D = x1.shape

    # Copy input to avoid modifying
    xt = x1.clone()  # (B, D)
    # Sample x_{t} from p_{t|1}(x_{t}|x_{1}) by masking
    mask_dims = torch.rand((B, D)).to(x1.device) < (1 - t[:, None])  # (B, D)
    xt[mask_dims] = mask_idx  # (B, D)

    # In case that pads should stay fixed during noising,
    # set all padded dims in x1 to pads in xt
    if pad_idx is not None:
        padded_dims = x1 == pad_idx  # (B, D)
        xt[padded_dims] = pad_idx  # (B, D)

    return xt


def predictor_loss_masking(
    predictor_log_prob_y: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    y: torch.Tensor,  # Shape: (B, ...)
    x1: torch.Tensor,  # Shape: (B, D)
    mask_idx: int,
    reduction: Literal["mean", "sum"] = "mean",
    pad_idx: Optional[int] = None,
    predict_on_x1: bool = False,
) -> torch.Tensor:
    """
    Computes loss for training a noisy predictor model with masked inputs.

    This function trains a predictor to estimate p(y|x,t) when x is partially masked
    according to time t.

    Args:
        predictor_log_prob_y: Function that takes (y: [B,...], x: [B,D], t: [B])
            and returns log p(y|x,t) of shape [B]
        y: Target values tensor of shape [batch_size, ...]
        x1: Input sequence tensor of shape [batch_size, sequence_length]
        mask_idx: Token index used for masking
        reduction: How to reduce the loss:
            - "mean": Average over batch (default)
            - "sum": Sum over batch
        pad_idx: Optional token index for padding that should be preserved during noising

    Returns:
        Negative log likelihood loss:
            - Scalar if reduction is "mean" or "sum"
            - Shape [batch_size] if reduction is "none"

    Raises:
        ValueError: If reduction is not "mean" or "sum"
    """
    B, D = x1.shape
    if not predict_on_x1:
        # Sample continuous time point
        t = torch.rand((B,)).to(x1.device)

        # Sample xt by masking x1 according to time t
        xt = sample_xt(x1, t, mask_idx, pad_idx)
        kwargs = {
            "y": y,
            "xt": xt,
            "t": t,
        }
    
    else:

        kwargs = {
            "y": y,
            "x1": x1,
        }

    # The model outputs logits over number of classes
    if reduction == "mean":
        return -torch.mean(predictor_log_prob_y(**kwargs))
    elif reduction == "sum":
        return -torch.sum(predictor_log_prob_y(**kwargs))

    # # The model outputs logits over number of classes
    # if reduction == "mean":
    #     return -torch.mean(predictor_log_prob_y(y, xt, t))
    # elif reduction == "sum":
    #     return -torch.sum(predictor_log_prob_y(y, xt, t))
    else:
        raise ValueError(
            "Input 'reduction' must be either 'sum' or 'mean', got '{reduction}' instead."
        )


def sample_pads_for_x0(x0, pad_idx, num_unpadded_freq_dict):
    """
    Sample pads for x0 (fully masked) and pad x0 with it.

    Args:
        x0 (torch.tensor): Torch tensor of shape (B, D) holding
            the noisy sampled discrete space values.
        pad_idx (int): Pad index.
        num_unpadded_freq_dict (dict): Discrete distribution of the number
            of unpadded tokens as dictionary mapping the number of unpadded
            tokens to their frequency in the form: {<#unpadded>: <frequency>}
            Example: {1: 10, 2: 5, 3: 3}

    Return:
        (torch.tensor): x0 with entries that have been padded.

    """
    # Extract variables from xt
    B = x0.shape[0]
    D = x0.shape[1]
    device = x0.device

    # Ensure that the dict-keys (i.e. "number of unpadded tokens") are integers
    num_unpadded_freq_dict = {
        int(num_unpadded): freq for num_unpadded, freq in num_unpadded_freq_dict.items()
    }

    # Generate a list with the 'number of unpadded token' values and an array with the associated probabilities
    # only including number of unpadded tokens in [0, D] because there are D tokens so that not more than D
    # tokens can be unpadded.
    num_unpadded_vals = [
        num_unpadded_val
        for num_unpadded_val in num_unpadded_freq_dict.keys()
        if num_unpadded_val <= D
    ] ## 3 to 100
    freq_num_unpadded = np.array(
        [num_unpadded_freq_dict[num_token] for num_token in num_unpadded_vals]
    )
    sum_freq_num_unpadded = np.sum(freq_num_unpadded)
    if 0 == sum_freq_num_unpadded:
        err_msg = f"Cannot compute 'number of unpadded tokens' probabilities because the 'number of unpadded tokens' frequencies sum to zero."
        raise ValueError(err_msg)
    elif sum_freq_num_unpadded < 0:
        err_msg = f"Cannot compute 'number of unpadded tokens' probabilities because the 'number of unpadded tokens' frequencies sum to something less than zero!"
        raise ValueError(err_msg)
    else:
        prob_num_unpadded = freq_num_unpadded / np.sum(
            sum_freq_num_unpadded
        )  # Normalize

    ## Sample the number of tokens
    # Step 1: Draw single sample (n=1) from Multinomial with the 'number of token' probabilities
    #         (prob_num_unpadded) for each batch point (size=B).
    #         np.random.multinomial returns a one-hot vector of len(pvals) per batch point.
    #         Argmax of each one hot vector returns a categorical corresponding to the index a
    #         certain 'number of token' in num_token_vals.
    num_unpadded_index_samples = np.argmax(
        np.random.multinomial(
            n=1, pvals=prob_num_unpadded, size=B
        ),  # (B, len(prob_num_unpadded))
        axis=-1,
    )  # (B, )

    # Step 2: Extract the 'number of unpadded' tokens corresponding to the sampled indices
    num_unpadded_samples = [
        num_unpadded_vals[num_unpadded_index]
        for num_unpadded_index in num_unpadded_index_samples
    ]  # '(B,)'

    ## Generate a 2D torch tensor of shape (B, D) that has True for the dimensions (second tensor axis)
    ## that should be padded for each point in the batch (first tensor axis)
    padded_dims = torch.ones((B, D), dtype=torch.bool, device=device)  # (B, D)
    for batch_pnt_index, num_unpadded_sample in enumerate(num_unpadded_samples):
        # 'padded_dims' is initialized with True entries, thus for the current batch point
        # (first tensor-axis) set the entries to False that should not be padded.
        # The non-added entries are the first 'num_unpadded_sample' ones, while the
        # tokens afterwards should be padded.
        padded_dims[batch_pnt_index, :num_unpadded_sample] = False

    # Pad the dimensions in xt that should be padded (for each batch point)
    # Remark: If we do not use xt.clone(), we get the following two warnings:
    #         (a) UserWarning: Use of index_put_ on expanded tensors is deprecated.
    #           Please clone() the tensor before performing this operation.
    #           This also applies to advanced indexing e.g. tensor[indices] = tensor
    #           (Triggered internally at ../aten/src/ATen/native/TensorAdvancedIndexing.cpp:716.)
    #         (b) UserWarning: Use of masked_fill_ on expanded tensors is deprecated.
    #           Please clone() the tensor before performing this operation.
    #           This also applies to advanced indexing e.g. tensor[mask] = scalar
    #           (Triggered internally at ../aten/src/ATen/native/TensorAdvancedIndexing.cpp:1914.)
    x0_padded = x0.clone()
    x0_padded[padded_dims] = pad_idx

    return x0_padded, padded_dims