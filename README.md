# <p align=center> :fire: `AR-1-to-3: Single Image to Consistent 3D Object via Next-View Prediction` :fire:</p>


![teaser_img](assets/teaser.png)
<div align="center">
  
  [[Paper](https://arxiv.org/pdf/2503.12929)] &emsp; [[Project Page](https://zhangxuying1004.github.io/projects/AR123/)] &emsp;  [[Jittor Version]()]&emsp; [[Demo]()]   <br>

</div>

If you have any questions about our work, feel free to contact us via zhangxuying1004@gmail.com.  
If our work is helpful to you or gives you some inspiration, please star this project and cite our paper. Thank you!


## 🚩 Todo List
- [x] Source code of AR123.
- [x] Evaluation code.
- [x] Training code.
- [x] Pretrained weights of AR123.
- [ ] Rendered dataset under the Zero123plus Setting.


## ⚙️ Setup
### 1. Dependencies and Installation
We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name ar123 python=3.10
conda activate ar123
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# Install other requirements
pip install -r requirements.txt
```
### 2. Downloading Datasets
We provide our rendered [objaverse subset]() under the Zero123++ configuration to facilitate reproducibility and further research.
Please download and place it into `zero123plus_renders`.


### 3. Downloading Checkpoints
Download [checkpoints]() and put them into `ckpts`.


## ⚡ Quick Start

### 1. Multi-View Synthesis
To synthesize multiple new perspective images based on a single-view image, please run:
``` 
CUDA_VISIBLE_DEVICES=0 python run.py --base configs/ar123_infer.yaml --input_path examples/c912d471c4714ca29ed7cf40bc5b1717.png --mode itomv
```

### 2. MV-to-3D Generation 
To generate 3D asset based on the synthesized multiple new views, please run:
``` 
CUDA_VISIBLE_DEVICES=0 python run.py --base configs/ar123_infer.yaml --input_path examples/c912d471c4714ca29ed7cf40bc5b1717.png --mode mvto3d
```

### 3. Image-to-3D Generation
You can also directly obtain 3D asset based on a single-view image by running:
``` 
CUDA_VISIBLE_DEVICES=0 python run.py --base configs/ar123_infer.yaml --input_path examples/c912d471c4714ca29ed7cf40bc5b1717.png --mode ito3d
```


## 🚀 Training

To train the default model, please run:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
    --base configs/ar123_train.yaml \
    --gpus 0,1,2,3,4,5,6,7 \
    --num_nodes 1
```

参数说明：
- `--base`: Path to configuration file
- `--gpus`: GPU device ID in use
- `--num_nodes`: Node number in use


## 🤖 Evaluation
### 1. 2D Evaluation (PSNR, SSIM, Clip-Score, LPIPS)
Please refer to `eval_2d.py`.

### 2. 3D Evaluation (Chamfer Distance, F-Score)
Please refer to `eval_3d.py`.


## 📦 Mesh Rendering
For beginners not familiar with the Blender software, we also provide mesh rendering codes that can run automatically on the cmd.
Please refer to the [render README](render/RENDER.md) for more details.



## 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [InstantMesh](https://github.com/TencentARC/InstantMesh)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [Zero123](https://github.com/cvlab-columbia/zero123)

In addition, we would like to express our sincere thanks to Jiale Xu for his invaluable assistance here.
