# Symbolic Music Generation (TreeG)

This repository contains the codebase for the symbolic music generation task (continuous diffusion with non-differentiable objectives) as presented in the paper:

[Training-Free Guidance Beyond Differentiability: Scalable Path Steering with Tree Search in Diffusion and Flow Models](https://arxiv.org/abs/2502.11420). 


## Set up the environment
- Create conda virtual environment via: `conda env create -f environment.yml`
- Activating virtual env: `conda activate guided`

## Download Pretrained Checkpoints
this is from paper [Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion](https://arxiv.org/abs/2402.14285). 
- Download the pretrained [VAE](https://huggingface.co/yjhuangcd/rule-guided-music/tree/main/trained_models/VAE) checkpoint.
- Download the pretrained [Diffusion Model](https://huggingface.co/yjhuangcd/rule-guided-music/tree/main/trained_models/diffusion).


## TreeG Generation

checkpoint path <diffusion_ckpt_path>, <vae_path>, set <save_dir>
for example, for note density task, set corresponding data_dir and config_path.


TreeG-SD: 

```
python sample.py \
  --active_size 2 \
  --branch_size 8 \
  --search_type "TreeG-SD" \
  --timestep_respacing "ddim1000" \
  --gs_n_iter 2 \
  --gs_pred_xstart_scale 0.5 \
  --gs_guidance_rate 1 \
  --seed 0 \
  --data_dir data/gen_target/note_density_target.csv \
  --save_dir <save_dir> \
  --config_path scripts/configs/nd_treeg_sd.yml \
  --batch_size 200 \
  --num_samples 1 \
  --model DiTRotary_XL_8 \
  --model_path <diffusion_ckpt_path> \
  --image_size 128 16 \
  --in_channels 4 \
  --scale_factor 1.2465 \
  --class_cond True \
  --num_classes 3 \
  --class_label 1 \
  --vae_path <vae_path>
```

where refers to Appendix F.1 TreeG-SD Setup section, gs_n_iter is N_iter, gs_pred_xstart_scale is hyperparameter mulitply stepsize ρt.


TreeG-SC

```
python sample.py \
  --active_size 2 \
  --branch_size 8 \
  --search_type "TreeG-SC" \
  --timestep_respacing "ddim1000" \
  --seed 0 \
  --data_dir data/gen_target/note_density_target.csv \
  --save_dir <save_dir> \
  --config_path scripts/configs/nd_treeg_sc.yml \
  --batch_size 200 \
  --num_samples 1 \
  --model DiTRotary_XL_8 \
  --model_path <diffusion_ckpt_path> \
  --image_size 128 16 \
  --in_channels 4 \
  --scale_factor 1.2465 \
  --class_cond True \
  --num_classes 3 \
  --class_label 1 \
  --vae_path <vae_path>
```
                                    

## References
This repository is based on 
- [Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion](https://github.com/yjhuangcd/rule-guided-music).
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion).
- The VAE architecture is modified upon [taming-transformers](https://github.com/CompVis/taming-transformers). 
- The DiT architecture is modified upon [DiT](https://github.com/facebookresearch/DiT).
- Music evaluation code is adapted from [mgeval](https://github.com/RichardYang40148/mgeval) and [figaro](https://github.com/dvruette/figaro).
- MIDI to piano roll representation is adapted from [pretty_midi](https://github.com/craffel/pretty-midi).




Please consider citing the following paper when using our code for your application.

```bibtex
@article{guo2025training,
  title={Training-free guidance beyond differentiability: Scalable path steering with tree search in diffusion and flow models},
  author={Guo, Yingqing and Yang, Yukang and Yuan, Hui and Wang, Mengdi},
  journal={arXiv preprint arXiv:2502.11420},
  year={2025}
}
```
