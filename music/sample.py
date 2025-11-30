"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random
import time
import pickle

import torch
import numpy as np
import torch as th
import pandas as pd
import torch.distributed as dist
import torch.nn.functional as F
import multiprocessing

from guided_diffusion import dist_util, midi_util, logger
from guided_diffusion.dit import DiT_models
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_diffusion,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_treesearch_diffusion,
)
from guided_diffusion.gaussian_diffusion import _encode, _extract_rule, _decode
from guided_diffusion.pr_datasets_all import load_data
from load_utils import load_model
import diff_collage as dc
from guided_diffusion.condition_functions import (
    model_fn, dc_model_fn, composite_nn_zt, composite_rule)
from functools import partial
import matplotlib.pyplot as plt

from music_rule_guidance.rule_maps import FUNC_DICT, LOSS_DICT
from music_rule_guidance.piano_roll_to_chord import IND2KEY



plt.rcParams["figure.figsize"] = (20, 3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    


    args = create_argparser().parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    root_dir = 'cond_demo/'
    if 'cond_table/' in args.config_path:
        args.dir = root_dir + os.path.splitext(args.config_path.split('cond_table/')[-1])[0] + f'_cls_{args.class_label}'
    else:
        args.dir = root_dir + os.path.splitext(args.config_path.split(root_dir)[-1])[0] + f'_cls_{args.class_label}'

    comm = dist_util.setup_dist(port=args.port)
    logger.configure(args=args, comm=comm)
    config = midi_util.load_config(args.config_path)

    


    logger.log("creating model and diffusion...")
    model = DiT_models[args.model](
        input_size=args.image_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma,
    )
    

    diffusion = create_treesearch_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        timestep_respacing=args.timestep_respacing,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        active_size=args.active_size,
        branch_size=args.branch_size,
        search_type=args.search_type,
    )



    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=False
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # create embed model
    if args.vae is not None:
        embed_model = load_model(args.vae, args.vae_path)
        embed_model.to(dist_util.dev())
        embed_model.eval()
    else:
        embed_model = None

    cond_fn_config = config.guidance.cond_fn
    if config.guidance.nn:
        logger.log("loading classifier...")
        classifier_config = cond_fn_config.classifiers
        num_classifiers = len(classifier_config.names)
        classifiers = []
        for i in range(num_classifiers):
            classifier = DiT_models[classifier_config.names[i]](
                # classifier trained on latents, so has the same img size as diffusion
                input_size=args.image_size,
                in_channels=args.in_channels,
                num_classes=classifier_config.num_classes[i],
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(classifier_config.paths[i], map_location="cpu")
            )
            classifier.to(dist_util.dev())
            classifier.eval()
            classifiers.append(classifier)

    if cond_fn_config is not None:
        if config.guidance.nn:
            cond_fn_used = partial(composite_nn_zt, fns=cond_fn_config.fns,
                                   classifier_scales=cond_fn_config.classifier_scales,
                                   classifiers=classifiers, rule_names=cond_fn_config.rule_names)
        else:
            cond_fn_used = partial(composite_rule, fns=cond_fn_config.fns,
                                   classifier_scales=cond_fn_config.classifier_scales,
                                   rule_names=cond_fn_config.rule_names)
    else:
        cond_fn_used = None

    if config.sampling.diff_collage:
        def eps_fn(x, t, y=None):
            # since our backbone takes 128x16 as input
            return model(x.permute(0, 1, 3, 2), t, y=y).permute(0, 1, 3, 2)

        # circle need one more num_img than linear
        img_shape = (args.in_channels, args.image_size[1], args.image_size[0])  # 4 x 16 x 128
        if config.dc.type == 'circle':
            worker = dc.CondIndCircle(img_shape, eps_fn, config.dc.num_img + 1, overlap_size=config.dc.overlap_size)
        else:
            worker = dc.CondIndSimple(img_shape, eps_fn, config.dc.num_img, overlap_size=config.dc.overlap_size)
        model_long_fn = worker.eps_scalar_t_fn
        gen_shape = (args.batch_size, worker.shape[0], worker.shape[2], worker.shape[1])
        model_fn_used = partial(dc_model_fn, model=model_long_fn, num_classes=args.num_classes,
                                class_cond=args.class_cond, cfg=args.cfg, w=args.w)
    else:
        gen_shape = (args.batch_size, args.in_channels, args.image_size[0], args.image_size[1])
        model_fn_used = partial(model_fn, model=model, num_classes=args.num_classes,
                                class_cond=args.class_cond, cfg=args.cfg, w=args.w)

    target_rules = vars(config.target_rules)
    source = 'given'
    # see if target rules are given, if not, extract from dataset
    for key, val in target_rules.items():
        if val is None:
            source = 'dataset'
        break

    if source == 'dataset':
        if 'vertical_nd' in target_rules.keys():
            # create a new dummy rule name and delete the old names
            target_rules['note_density'] = None
            target_rules.pop('vertical_nd')
            target_rules.pop('horizontal_nd')
        val = get_rule_target(data_path=args.data_dir, device=dist_util.dev())
        target = dict()
        for k, _ in target_rules.items():
            target[k] = val
        model_kwargs = {"rule": target}



    else:
        for key, val in target_rules.items():
            if 'vertical_nd' in key:
                # to make vertical and horizontal nd to be of similar scale
                if '_hr_' in key:
                    str_hr_scale = key.split('_hr_')[-1]
                    horizontal_scale = int(str_hr_scale)
                    horizontal_nd = [x / horizontal_scale for x in target_rules[f'horizontal_nd_hr_{str_hr_scale}']]
                    target_rules[f'note_density_hr_{str_hr_scale}'] = target_rules[key] + horizontal_nd
                else:
                    horizontal_scale = 5
                    horizontal_nd = [x / horizontal_scale for x in target_rules['horizontal_nd']]
                    target_rules['note_density'] = target_rules[key] + horizontal_nd

                target_rules.pop(key)
                target_rules.pop(key.replace('vertical', 'horizontal'))
                break

        for key, val in target_rules.items():
            val = th.tensor(val, device=dist_util.dev())
            if key == 'pitch_hist':
                val = val / (th.sum(val) + 1e-12)
            target_rules[key] = val
        model_kwargs = {"rule": {k: v.repeat(args.batch_size, 1) for k, v in target_rules.items()}}

    if args.class_cond:
        # only generate one class
        classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=th.int) * args.class_label
        model_kwargs["y"] = classes

    save_dir = args.save_dir if args.save_dir else logger.get_dir()
    search_type_dir = args.search_type if args.search_type else 'none'
    same_name = f"seed_{args.seed}_{args.timestep_respacing}"
    if args.search_type == 'TreeG-SD':
        same_name += f"_active_size_{args.active_size}_branch_size_{args.branch_size}_n_iter_{args.gs_n_iter}_xscale_{args.gs_pred_xstart_scale}_gs_rate_{args.gs_guidance_rate}"
    
    elif args.search_type == 'TreeG-SC':
        same_name += f"_active_size_{args.active_size}_branch_size_{args.branch_size}"
    save_dir = os.path.join(save_dir, search_type_dir, same_name)
    os.makedirs(os.path.expanduser(save_dir), exist_ok=True)

    ddim_stochastic = partial(diffusion.ddim_sample_loop, eta=args.eta)
    sample_fn = (
        diffusion.p_sample_loop if not config.sampling.use_ddim else ddim_stochastic
    )

    logger.log("sampling...")
    count_samples = 0
    all_results = pd.DataFrame()

    logger.log("active set size: ", args.active_size)
    logger.log("branch size: ", args.branch_size)
    logger.log("search method: ", args.search_type)

    
    # set parameters for gs
    if config.guidance.gs:
        config.guidance.gs.pred_xstart_scale = args.gs_pred_xstart_scale
        config.guidance.gs.guidance_rate = args.gs_guidance_rate
        config.guidance.gs.n_iter = args.gs_n_iter
        logger.log("guided x start sampling...")
        logger.log(f"n_iter: {args.gs_n_iter},"
                     f"pred_xstart_scale: {args.gs_pred_xstart_scale}, guidance_rate: {args.gs_guidance_rate}")
       
        config.guidance.gs.dsg = args.gs_dsg
        logger.log(f"use dsg: {config.guidance.gs.dsg}")


    # add active set dim
    gen_shape = (gen_shape[0]*args.active_size, *gen_shape[1:])
    model_kwargs_active = dict()
    model_kwargs_active["rule"] = dict()
    cond = model_kwargs["y"]
    cond = cond.repeat_interleave(args.active_size)
    model_kwargs_active["y"] = cond

    for k, v in model_kwargs["rule"].items():
        model_kwargs_active["rule"][k] = v.repeat_interleave(args.active_size, dim=0)
    


    
    while count_samples < args.num_samples:
        start_time = time.time()
        sample = sample_fn(
            model_fn_used,
            gen_shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs_active,
            device=dist_util.dev(),
            cond_fn=cond_fn_used,
            # None for NN(z_0), embed_model for rule(decoder(z_0))
            embed_model=embed_model if config.guidance.vae else None,
            scale_factor=args.scale_factor,
            guidance_kwargs=config.guidance,
            scg_kwargs=vars(config.scg) if config.guidance.scg else None,
            t_end=config.sampling.t_end,
            record=args.record,
            progress=True
        )
        
        # decode sample
        sample = decode_sample(sample, batch_size=1, embed_model=embed_model, scale_factor=args.scale_factor)

        arr = sample.cpu().numpy()
        arr = arr.transpose(0, 3, 1, 2)


        # test distance between generated samples and target
        generated_samples = th.from_numpy(arr) / 63.5 - 1

        # best of N
        if args.active_size > 1:
            generated_samples = get_best_of_active_set(generated_samples, args.active_size, model_kwargs_active["rule"])

        
       
        end_time = time.time()
        used_time = end_time - start_time
        logger.log(f"Time used: {used_time:.6f}s")

        start_time_eval = time.time()

        for _ in range(1):
            results = midi_util.eval_rule_loss(generated_samples, model_kwargs["rule"])

        end_time_eval = time.time()
        used_time_eval = (end_time_eval - start_time_eval) / 1
        logger.log(f"Time used for evaluation: {used_time_eval:.6f}s")

        all_results = pd.concat([all_results, results], ignore_index=True)
        ## print results
        print("active set size: ", args.active_size)
        print("branch size: ", args.branch_size)
        print("search method: ", args.search_type)
        if config.guidance.gs:
            #print("guided x start")
            print(f"n iter: {args.gs_n_iter}, pred_xstart_scale: {args.gs_pred_xstart_scale}, guidance_rate: {args.gs_guidance_rate}")

        print('Results:')
        for key, val in all_results.items():
            print(f'{key}: {val}')

        # save samples
        if args.save_files:
            arr = (generated_samples + 1) * 63.5
            arr = arr.cpu().numpy()

            sample_dir = os.path.join(save_dir, f'samples')
            os.makedirs(sample_dir, exist_ok=True)

            if args.class_cond:
                label_arr = classes.cpu().numpy()
                midi_util.save_piano_roll_midi(arr, sample_dir, args.fs, y=label_arr, save_ind=count_samples)
            else:
                midi_util.save_piano_roll_midi(arr, sample_dir, args.fs, save_ind=count_samples)

        # save every step
        if args.save_files:
            all_results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
        count_samples += args.batch_size





    if args.save_files:
        all_results.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
        # Create the DataFrame for loss_stats
        loss_columns = [col for col in all_results.columns if '.loss' in col]
        rows = []
        for col in loss_columns:
            rows.append({'Attr': col, 'Mean': all_results[col].mean(), 'Std': all_results[col].std()})

        loss_stats = pd.DataFrame(rows, columns=['Attr', 'Mean', 'Std'])
        loss_stats.to_csv(os.path.join(save_dir, 'summary.csv'))
        print(loss_stats)

   
        
        loss_stats.to_csv(os.path.join(save_dir, 'summary_all.csv'))
        print('Summary:')
        print(loss_stats)

    if args.record:
        # import pickle
        with open(os.path.join(save_dir, 'log_probs.pkl'), 'wb') as f:
            pickle.dump(diffusion.log_probs, f)
        with open(os.path.join(save_dir, 'loss_std.pkl'), 'wb') as f:
            pickle.dump(diffusion.loss_std, f)
        with open(os.path.join(save_dir, 'loss_range.pkl'), 'wb') as f:
            pickle.dump(diffusion.loss_range, f)
        with open(os.path.join(save_dir, 'each_loss.pkl'), 'wb') as f:
            pickle.dump(diffusion.each_loss, f)

        midi_util.plot_record(diffusion.log_probs, 'log_prob', save_dir)
        midi_util.plot_record(diffusion.loss_std, 'loss_std', save_dir)
        midi_util.plot_record(diffusion.loss_range, 'loss_range', save_dir)

        if len(diffusion.inter_piano_rolls) > 0:
            diffusion.inter_piano_rolls.append(th.from_numpy(arr))
            inter_piano_rolls = th.concat(diffusion.inter_piano_rolls, dim=0)
            save_dir_inter = os.path.join(save_dir, 'inter')
            os.makedirs(save_dir_inter, exist_ok=True)
            midi_util.save_piano_roll_midi(inter_piano_rolls.numpy(), save_dir=save_dir_inter, fs=args.fs)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        project="music-sampling",
        dir="",
        data_dir="",  # use to load in val data to extract rule
        config_path="",
        model="DiTRotary_XL_8",  # DiT model names
        model_path="",
        vae="kl/f8-all-onset",
        vae_path="taming-transformers/checkpoints/all_onset/epoch_14.ckpt",
        clip_denoised=False,
        num_samples=128,
        batch_size=16,
        scale_factor=1.,
        fs=100,
        num_classes=0,
        class_label=1,  # class to generate
        cfg=False,
        w=4.,  # for cfg
        classifier_scale=1.0,
        record=False,
        save_files=True,
        training=False,  # not training, so don't need to create more folders than needed
        deterministic=False,  # whether to use the same rule everytime
        port=None,
        seed=0,
        # gs_num_samples=16,
        # gs_temperature=1.0,
        gs_pred_xstart_scale=1.0,
        gs_guidance_rate=1.0,
        gs_n_iter=1,
        # gs_aggr="argmax",
        gs_dsg=True,
        timestep_respacing="ddim50",
        eta=1.0,
        save_dir = '',
        active_size=1,
        branch_size=8,
        search_type=None,
        t_start=750,
        interval=1,
        time_path=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def get_rule_target(data_path, device):
    data = pd.read_csv(data_path)
    val = data.iloc[:,0]
    val = val.apply(eval)
    val = torch.tensor(val.to_list(), device=device)
    print('target rule shape:', val.shape)
    return val

def decode_sample(sample, batch_size=1, embed_model=None, scale_factor=1.):
    num_samples = sample.size(0)  # Number of samples in the tensor
    processed_samples = []

    for i in range(0, num_samples, batch_size):
        # Slice the batch
        batch = sample[i:i+batch_size]
        
        # Process the smaller batch
        try:
            processed_batch = midi_util.decode_sample_for_midi(batch, embed_model=embed_model,
                                                  scale_factor=scale_factor, threshold=-0.95)
            processed_samples.append(processed_batch)
        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")

    # Combine all processed results
    final_result = torch.cat(processed_samples, dim=0)  # Adjust dim based on your data format
    return final_result
        

def get_best_of_active_set(samples, active_size, target_rule):
    batch_size = samples.shape[0]
    total_loss = 0
    for rule_name, rule_target in target_rule.items():
        rule_target_list = rule_target.tolist()
        if batch_size == 1:
            rule_target_list = [rule_target_list]
        # results[rule_name + '.target_rule'] = rule_target_list
        rule_target = rule_target.to(samples.device)
        if 'chord' in rule_name:
            gen_rule, key, corr = FUNC_DICT[rule_name](samples, return_key=True)
            key_strings = [IND2KEY[key_ind] for key_ind in key]
            loss = LOSS_DICT[rule_name](gen_rule, rule_target)
            # mean_loss, std_loss, gen_rule, loss = loss.mean(), loss.std(), gen_rule.tolist(), loss.tolist()
            if batch_size == 1:
                gen_rule = [gen_rule]
            
        else:
            gen_rule = FUNC_DICT[rule_name](samples)
            loss = LOSS_DICT[rule_name](gen_rule, rule_target)
            # mean_loss, std_loss, gen_rule, loss = loss.mean(), loss.std(), gen_rule.tolist(), loss.tolist()
            if batch_size == 1:
                gen_rule = [gen_rule]
            

        total_loss += loss
    

    print('loss before best-of-N:')
    print('mean:', total_loss.mean().item())
    print('std:', total_loss.std().item())

    total_loss = total_loss.view(-1, active_size)
    best_idx = total_loss.argmin(dim=1)
    samples = samples.view(-1, active_size, *samples.shape[1:])
    best_samples = samples[range(samples.shape[0]), best_idx]

    return best_samples




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
