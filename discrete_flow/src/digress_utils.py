import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Callable, Literal

import src.fm_utils as fm_utils


def d3pm_sample_xt(
    x1: torch.Tensor,
    t: torch.Tensor,
    mask_idx: int,
    timesteps: int,
    pad_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Samples a noised state xt by masking x1 according to time t.

    Args:
        x1: Input sequence tensor of shape (B, D)
        t: Time values tensor of shape (B,)
        mask_idx: Index representing the mask token
        timesteps: The number of discrete timesteps used in D3PM training
        pad_idx: Optional index representing padding token that should be preserved

    Returns:
        Noised sequence tensor of shape (B, D)
    """
    if (t < 1).any() or (t > timesteps).any():
        raise ValueError(f"t needs to be 1 <= t <= {timesteps}")

    B, D = x1.shape

    # Copy input to avoid modifying
    xt = x1.clone()  # (B, D)
    # Sample x_{t} from p_{t|1}(x_{t}|x_{1}) by masking
    mask_dims = torch.rand((B, D)).to(x1.device) < (t.view(B, 1) / timesteps)  # (B, D)
    xt[mask_dims] = mask_idx  # (B, D)

    # In case that pads should stay fixed during noising,
    # set all padded dims in x1 to pads in xt
    if pad_idx is not None:
        padded_dims = x1 == pad_idx  # (B, D)
        xt[padded_dims] = pad_idx  # (B, D)

    return xt


def d3pm_loss_masking(
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x1: torch.Tensor,
    mask_idx: int,
    reduction: Literal["mean", "sum"] = "mean",
    pad_idx: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    timesteps: int = 1000,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    D3PM CE loss for the masking interpolate

    Args:
        denoising_model (function): Can treat model(x, t) with x discrete
        x1 (torch.tensor): Batched states of shape (B, D).
        mask_idx (int): Index of mask state.
        reduction (str): 'none', 'mean', 'sum' used as input to
            'torch.nn.functional.cross_entropy'.
        pad_idx (int): Index of pad state.
        loss_mask (torch.tensor): Mask tensor of shape (B, D) with
            True for positions that are included in the loss calculation
            and False for positions that are not.

    """
    B, D = x1.shape
    # <mask> is the last index
    # t = torch.rand((B,)).to(x1.device)
    # Sample discrete time points
    t = torch.randint(low=1, high=timesteps + 1, size=(B,)).float().to(x1.device)

    # Corrupt with masks, assume 0, 1, ..., S-2 are the valid values and S-1 represents MASK
    # xt = x1.clone()
    # xt[torch.rand((B, D)).to(x1.device) < (1 - t[:, None])] = mask_idx
    xt = d3pm_sample_xt(x1, t, mask_idx, timesteps, pad_idx)

    # Calculate the logits
    # logits = denoising_model(xt, t) # (B, D, S-1) # TODO: Should we specify S or S-1 output
    # Feed in the normalized time
    logits = denoising_model(
        xt, t / timesteps
    )  # (B, D, S-1) # TODO: Should we specify S or S-1 output

    # Determine the positions that should be excluded from the code
    exclude = xt != mask_idx  # Exclude all positions that are not masked
    if loss_mask is not None:
        exclude = torch.logical_or(
            exclude, ~loss_mask
        )  # In addition, exclude all positions that are False in loss_mask

    x1 = (
        x1.clone()
    )  # Copy, so that we are not overwriting the (external) passed x1 in the line below
    x1[exclude] = -1  # don't compute the loss on dimensions that are already revealed

    # x1[xt != mask_idx] = -1 # don't compute the loss on dimensions that are already revealed
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        x1,
        reduction=reduction,
        ignore_index=-1,
        label_smoothing=label_smoothing,
    )
    return loss


def d3pm_predictor_loss_masking(
    predictor_log_prob_y: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    y: torch.Tensor,
    x1: torch.Tensor,
    mask_idx: int,
    timesteps: int,
    reduction: Literal["mean", "sum"] = "mean",
    pad_idx: Optional[int] = None,
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
        timesteps: The number of discrete timesteps used in D3PM training
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
    # Sample random discrete time point
    t = torch.randint(low=1, high=timesteps + 1, size=(B,)).float().to(x1.device)

    # Sample xt by masking x1 according to time t
    xt = d3pm_sample_xt(x1, t, mask_idx, timesteps, pad_idx)

    # The model outputs logits over number of classes
    if reduction == "mean":
        return -torch.mean(predictor_log_prob_y(y, xt, t / timesteps))
    elif reduction == "sum":
        return -torch.sum(predictor_log_prob_y(y, xt, t / timesteps))
    else:
        raise ValueError(
            "Input 'reduction' must be either 'sum' or 'mean', got '{reduction}' instead."
        )


def d3pm_sampling(
    num_samples: int,
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    S: int,
    D: int,
    device: torch.device,
    timesteps: int = 1000,
    mask_idx: Optional[int] = None,
    pad_idx: Optional[int] = None,
    predictor_log_prob: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    guide_temp: float = 1.0,
    batch_size: int = 500,
    x1_temp: float = 1.0,
    num_unpadded_freq_dict: Optional[dict[int, float]] = None,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Generates samples with the discrete time discrete diffusion model (D3PM)
    with optional guidance.
    This is a wrapper function that generates samples in batches for memory efficiency.

    Args:
        denoising_model: Model that takes (x_t: [B,D], t: [B]) and returns logits [B,D,S]
        S: Size of categorical state space (vocabulary size)
        D: Dimension of each sample (sequence length)
        device: Device to run generation on
        timesteps: The number of discrete time step used in D3PM training
        mask_idx: Token index used for masking. Defaults to S-1
        pad_idx: Optional token index used for padding
        predictor_log_prob: Optional predictor function for guided sampling that takes (x,t)
            and returns log p(y|x,t) of shape [B]
        guide_temp: Temperature for guidance (1 / \gamma). Lower = stronger guidance
        batch_size: Number of samples to generate in parallel
        x1_temp: Temperature for softmax of model logits
        num_unpadded_freq_dict: Optional dict mapping num unpadded tokens to frequencies
        eps: Small constant for numerical stability

    Returns:
        numpy.ndarray: Generated samples of shape [batch_size, D]
    """
    # Inform the user
    print(
        f"Generating {num_samples} samples: #timesteps={timesteps}, guide_temp={guide_temp}"
    )

    if batch_size > num_samples:
        batch_size = num_samples

    counter = 0
    samples = []
    while True:
        # Call the wrapped sampling function for the current batch
        x1 = _d3pm_sampling(
            denoising_model=denoising_model,
            batch_size=batch_size,
            S=S,
            D=D,
            device=device,
            timesteps=timesteps,
            mask_idx=mask_idx,
            pad_idx=pad_idx,
            predictor_log_prob=predictor_log_prob,
            guide_temp=guide_temp,
            x1_temp=x1_temp,
            num_unpadded_freq_dict=num_unpadded_freq_dict,
            eps=eps,
        )
        samples.append(x1)
        counter += batch_size
        print(f"{counter} out of {num_samples} generated")
        if counter >= num_samples:
            break
    samples = np.concatenate(samples, axis=0)[:num_samples]
    return samples


def _d3pm_sampling(
    denoising_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: int,
    S: int,
    D: int,
    device: torch.device,
    timesteps: int = 1000,
    mask_idx: int = None,
    pad_idx: int = None,
    predictor_log_prob: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    guide_temp: float = 1.0,
    x1_temp: float = 1.0,
    num_unpadded_freq_dict: Optional[dict[int, float]] = None,
    eps: float = 1e-10,
):
    """
    Generates samples with the discrete time discrete diffusion model (D3PM)
    with optional guidance.
    This is a wrapper function that generates samples in batches for memory efficiency.

    Args:
        denoising_model: Model that takes (x_t: [B,D], t: [B]) and returns logits [B,D,S]
        batch_size: Number of samples to generate in parallel
        S: Size of categorical state space (vocabulary size)
        D: Dimension of each sample (sequence length)
        device: Device to run generation on
        timesteps: The number of discrete time step used in D3PM training
        mask_idx: Token index used for masking. Defaults to S-1
        pad_idx: Optional token index used for padding
        predictor_log_prob: Optional predictor function for guided sampling that takes (x,t)
            and returns log p(y|x,t) of shape [B]
        guide_temp: Temperature for guidance (1 / \gamma). Lower = stronger guidance
        x1_temp: Temperature for softmax of model logits
        num_unpadded_freq_dict: Optional dict mapping num unpadded tokens to frequencies
        eps: Small constant for numerical stability

    Returns:
        numpy.ndarray: Generated samples of shape [batch_size, D]
    """
    if mask_idx is None:
        mask_idx = S - 1

    B = batch_size

    # Sample initial xt
    xt = mask_idx * torch.ones((B, D), dtype=torch.long, device=device)

    # Create discrete time grids
    ts = np.arange(timesteps, 0, -1)
    mask_one_hot = torch.zeros((S,), device=device)
    mask_one_hot[mask_idx] = 1.0

    # Treat the case where fixed pads should be used
    if pad_idx is not None:
        pad_one_hot = torch.zeros((S,), device=device)
        pad_one_hot[pad_idx] = 1.0

        # If 'num_unpadded_freq_dict' is not None,
        # sample pads for x0 (=xt at time t=0) and pad xt
        # overwriting the current xt
        if num_unpadded_freq_dict is not None:
            xt = fm_utils.sample_pads_for_x0(xt, pad_idx, num_unpadded_freq_dict)

    # Iterate over discrete time grids
    for t in tqdm(ts):
        # Get denoising model prediction
        logits = denoising_model(
            xt, t / timesteps * torch.ones((B,), device=device)
        )  # (B, D, S)
        # Since we explicitly normalize, and the probability of staying in mask
        # state is taken care of later (also no remasking)
        # we should set the probability to mask to 0
        logits[:, :, mask_idx] = -1e4
        x0_probs = F.softmax(logits / x1_temp, dim=-1)  # (B, D, S)
        xt_is_mask = (xt == mask_idx).view(B, D, 1).float()

        # Compute probs for unmasking and staying in mask
        step_probs = (1 / t) * x0_probs + (1 - 1 / t) * mask_one_hot.view(
            1, 1, -1
        )  # (B, D, S)

        # Modify the probabilities in case that the padded states are fixed
        if pad_idx is not None:
            # Step 1: Suppress unpadded -> padded transitions
            # When fixing the padded states, do not allow transitions from unpadded states to
            # padded states by setting the probabilities of these transitions to zero
            step_probs *= 1 - pad_one_hot.view(1, 1, -1)

            # Step 2: Suppress padded -> unpadded transitions
            # When fixing the padded states, do not allow transitions from padded states to
            # unpadded states by setting the probabilities of these transitions to zero (2a) and
            # the probability of a padded state to itself (i.e. padded to padded) to one (2b).
            # Step 2a:
            xt_is_pad = (xt == pad_idx).view(B, D, 1).float()
            step_probs *= 1 - xt_is_pad
            # Step 2b:
            step_probs += xt_is_pad * pad_one_hot.view(1, 1, -1)

        # Perform digress updates
        if predictor_log_prob is not None:
            step_probs = get_digress_adjusted_probs(
                predictor_log_prob,
                xt,
                t,
                step_probs,
                S,
                guide_temp=guide_temp,
                timesteps=timesteps,
                eps=eps,
            )
        new_xt = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(B, D)

        # If current state is masked, assign states to be the newly sampled states
        xt = xt * (1 - xt_is_mask[:, :, 0].long()) + new_xt * xt_is_mask[:, :, 0].long()

    return xt.detach().cpu().numpy()


def get_digress_adjusted_probs(
    predictor_log_prob: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xt: torch.Tensor,  # Shape: (B, D)
    t: float,
    step_probs: torch.Tensor,  # Shape: (B, D, S)
    S: int,
    guide_temp: int = 1,
    log_prob_ratio_cutoff: float = 80,
    timesteps: int = 1000,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Computes guide-adjusted probabilities with the DiGress updates

    Args:
        predictor_log_prob (callable): Function that takes (x,t) and returns log p(y|x,t)
        xt (torch.Tensor): Current states of shape (B, D)
        t (float): Current time
        step_probs (torch.Tensor): Unconditional transition probabilities of shape (B, D, S)
        S (int): Size of categorical state space
        guide_temp (float, optional): Guidance temperature. Defaults to 1.
        log_prob_ratio_cutoff (float, optional): Maximum value for log ratios. Defaults to 80.
        timesteps (int): The number of timesteps used in D3PM training
    Returns:
        torch.Tensor: Guide-adjusted probabilities of shape (B, D, S)
    """
    B, D = xt.shape
    device = xt.device
    t = t * torch.ones((B,), device=device)

    # One-hot encode categorical data, shape (B, D, S)
    xt_ohe = F.one_hot(xt.long(), num_classes=S).to(torch.float)

    # \grad_{x_t}{log p(y|x_t)}(x), shape (B, D, S)
    with torch.enable_grad():
        xt_ohe.requires_grad_(True)
        # log p(y|x_t=x), shape (B,)
        log_prob_xt_ohe = predictor_log_prob(xt_ohe, t / timesteps)
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

    # Multiply the transition probability (already normalized)
    # elementwise with the guidance term
    # Shape (B, D, S)
    # Note:
    # 1. This modulates each dimension independently, which DiGress assumes
    # 2. This doesn't deal with normalization, we later need to normalize per dimension
    step_probs = torch.exp(torch.log(step_probs) + log_prob_ratio) + eps

    # Obtain normalized probabilities
    step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True)

    return step_probs
