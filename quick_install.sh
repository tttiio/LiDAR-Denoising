#!/bin/bash
# ============================================================
# 快速安装命令 (复制粘贴运行)
# ============================================================

# 1. 下载并安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 2. 创建环境
conda create -n denoise python=3.10 -y
conda activate denoise

# 3. 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 4. 安装其他依赖
pip install numpy scipy pyyaml matplotlib seaborn scikit-learn tqdm open3d

# 5. 编译扩展
cd /path/to/LiDAR-Denoising
python setup.py build_ext --inplace

# 6. 测试
python test_denoise.py
