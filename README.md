# UEPS: Robust and Efficient MRI Reconstruction

This is the official pytorch implementation of the paper 
["UEPS: Robust and Efficient MRI Reconstruction"](https://arxiv.org/abs/2603.18572).

*Xiang Zhou, Hong Shang, Zijian Zhan, Tianyu He, Jintao Meng, Dong Liang*

*Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences*

>  ### Abstract
> *Deep unrolled models (DUMs) have become the state of the art for accelerated MRI reconstruction, yet their robustness under domain shift remains a critical barrier to clinical adoption. In this work, we identify coil sensitivity map (CSM) estimation as the primary bottleneck limiting generalization. To address this, we propose UEPS, a novel DUM architecture featuring three key innovations: (i) an Unrolled Expanded (UE) design that eliminates CSM dependency by reconstructing each coil independently; (ii) progressive resolution, which leverages k-space-to-image mapping for efficient coarse-to-fine refinement; and (iii) sparse attention tailored to MRI's 1D undersampling nature. These physics-grounded designs enable simultaneous gains in robustness and computational efficiency. We construct a large-scale zero-shot transfer benchmark comprising 10 out-of-distribution test sets spanning diverse clinical shifts—anatomy, view, contrast, vendor, field strength, and coil configurations. Extensive experiments demonstrate that UEPS consistently and substantially outperforms existing DUM, end-to-end, diffusion, and untrained methods across all OOD tests, achieving state-of-the-art robustness with low-latency inference suitable for real-time deployment.*

## 💪 Get Started
### 🛠️ Installation

1. Create conda environment

```bash
conda create -n yourenvname python=3.12
conda activate yourenvname
```

2. Install pytorch

  Follow official pytorch installation for version of 2.9.1 or later.

3. Install other packages

```bash
pip install -r requirements.txt
```

### 📂 Data Preparation
We use the [fastMRI Brain](https://fastmri.med.nyu.edu/) dataset for training.
1. Download the dataset from the official website.
2. Organize the dataset directory as follows:
```text
/path/to/your/dataset/
└── fastmri_brain_mc/
    ├── multicoil_train/
    ├── multicoil_val/
    └── multicoil_test_full/
```
3. Update the data path in your configuration files (e.g., `configs/fastmri_UEPS.yaml`).

### 🚀 Training
Example of training with single GPU or Distributed Data Parallel (DDP)
```bash
cd tools
python -m accelerate.commands.launch --config_file "../configs/ddp.yaml" train.py --output_dir "../model_export/somename" --config "../configs/someconfig.yaml"
```

Example of training with Fully Sharded Data Parallel (FSDP):
```bash
cd tools
python -m accelerate.commands.launch --config_file "../configs/fsdp.yaml" train.py --output_dir "../model_export/somename" --config "../configs/someconfig.yaml"
```

### 📊 Evaluation
Example of evaluating a saved checkpoint on some testset
```bash
cd tools
CUDA_VISIBLE_DEVICES=0 nohup python test.py \
--config "../configs/someconfig.yaml" \
--ckpt_path "../checkpoints/ckpt_pick.pth" \
--data_dir "../demo_data" \
--output_dir "../model_export/somename_eval" \
> somename_eval.log &
```

### 🔗 Pre-trained Models and Demo Data

We host our pre-trained model weights and sample demo data on Zenodo to facilitate easy reproduction.

1. Download the weights (`ckpt_pick.pth`) and place it in the `checkpoints/` directory.
2. Download the demo data (`demo_data.zip`) and extract it into the `demo_data/` directory.

[📥 Download weights and data from Zenodo](https://zenodo.org/records/19093602)

## ⭐ Citation
If you find UEPS useful for your research or projects, we would greatly appreciate it if you could cite the following paper:
```bibtex
@misc{zhou2026uepsrobustefficientmri,
      title={UEPS: Robust and Efficient MRI Reconstruction}, 
      author={Xiang Zhou and Hong Shang and Zijian Zhan and Tianyu He and Jintao Meng and Dong Liang},
      year={2026},
      eprint={2603.18572},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2603.18572}, 
}
```
