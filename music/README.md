# Symbolic Music Generation (TreeG)

This repository contains the codebase for the symbolic music generation task (continuous diffusion with non-differentiable objectives) as presented in the paper: [Training-Free Guidance Beyond Differentiability: Scalable Path Steering with Tree Search in Diffusion and Flow Models](https://arxiv.org/abs/2502.11420). 


---

## 🛠️ Environment Setup

To get started, create and activate the required Conda environment:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate guided
```

---

## 💾 Pretrained Checkpoints

This project utilizes pretrained models from the paper [Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion](https://arxiv.org/abs/2402.14285).

Please download the following checkpoints and save them to your local directory:

* **VAE Checkpoint:** Download from [VAE](https://huggingface.co/yjhuangcd/rule-guided-music/tree/main/trained_models/VAE).
* **Diffusion Model Checkpoint:** Download from [Diffusion Model](https://huggingface.co/yjhuangcd/rule-guided-music/tree/main/trained_models/diffusion).

---

## 🚀 TreeG Generation

Below are the commands to run generation using **TreeG-SD** and **TreeG-SC**.

### Usage Notes

Before running the scripts, ensure you replace the following placeholders with your actual paths:

* `<diffusion_ckpt_path>`: Path to the downloaded Diffusion Model checkpoint.
* `<vae_path>`: Path to the downloaded VAE checkpoint.
* `<save_dir>`: Directory where the generated results will be saved.

**Note:** The examples below target the *Note Density* task. To change the task, update the `--data_dir` and `--config_path` arguments accordingly.


### TreeG-SD

Hyperparameters (explained in detail in **Appendix F.1** of the paper): 

* `--gs_n_iter`: \(N_{\text{iter}}\).
* `--gs_pred_xstart_scale`:Scaling factor applied to the step size (\(\rho_t\)).  


```bash
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


### TreeG-SC

```bash
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
                                    
## 📚 References

This repository extends and modifies  
[**Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion**](https://github.com/yjhuangcd/rule-guided-music).

Please consider citing the following paper when using our code:

```bibtex
@article{guo2025training,
  title={Training-free guidance beyond differentiability: Scalable path steering with tree search in diffusion and flow models},
  author={Guo, Yingqing and Yang, Yukang and Yuan, Hui and Wang, Mengdi},
  journal={arXiv preprint arXiv:2502.11420},
  year={2025}
}

