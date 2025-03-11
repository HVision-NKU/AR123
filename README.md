# <p align=center> :fire: `AR-1-to-3: Single Image to Consistent 3D Object via Next-View Prediction`</p>


![teaser_img](figs/teaser.png)
<div align="center">
  
  [[Paper](https://arxiv.org/pdf/2412.16919)] &emsp; [[Project Page](https://zhangxuying1004.github.io/projects/TAR3D/)] &emsp;  [[Jittor Version]()]&emsp; [[Demo]()]   <br>

</div>




## ⚙️ Setup
### 1. Dependencies and Installation
We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name tar3d python=3.10
conda activate tar3d
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



### 3. Downloading Checkpoints


## ⚡ Quick Start

### 1. Multi-View Synthesis

### 2. 3D Generation


## 💻 Training




## 💫 Evaluation
### 1. 2D Evaluation (PSNR, SSIM, Clip-Score, LPIPS)

### 2. 3D Evaluation (Chamfer Distance, F-Score)


## 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [InstantMesh](https://github.com/TencentARC/InstantMesh)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
