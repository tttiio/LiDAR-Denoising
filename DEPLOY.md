# 阿里云服务器部署指南

## 方式一：一键安装脚本

### 1. 上传项目到服务器

```bash
# 方式A: 使用 scp
scp -r LiDAR-Denoising user@server:/home/user/

# 方式B: 使用 rsync
rsync -avz LiDAR-Denoising user@server:/home/user/

# 方式C: 先上传压缩包
tar -czvf LiDAR-Denoising.tar.gz LiDAR-Denoising
scp LiDAR-Denoising.tar.gz user@server:/home/user/
# 服务器上解压
ssh user@server
tar -xzvf LiDAR-Denoising.tar.gz
```

### 2. 运行安装脚本

```bash
cd /home/user/LiDAR-Denoising
chmod +x install_server.sh
./install_server.sh
```

### 3. 等待安装完成

脚本会自动完成：
- 安装 Miniconda
- 创建 Python 环境
- 安装 PyTorch 和依赖
- 编译 CUDA 扩展
- 验证安装

---

## 方式二：手动安装（复制粘贴）

### 步骤 1: 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 步骤 2: 创建环境

```bash
conda create -n denoise python=3.10 -y
conda activate denoise
```

### 步骤 3: 安装 PyTorch

```bash
# 先查看 CUDA 版本
nvidia-smi

# CUDA 12.x
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.x
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

### 步骤 4: 安装依赖

```bash
pip install numpy scipy pyyaml matplotlib seaborn scikit-learn tqdm open3d
```

### 步骤 5: 编译扩展

```bash
cd /home/user/LiDAR-Denoising
python setup.py build_ext --inplace
```

### 步骤 6: 验证安装

```bash
python test_denoise.py
```

---

## 训练命令

```bash
# 激活环境
conda activate denoise

# 单 GPU 训练
python train_denoise.py --config config/config_denoise_semanticstf

# 多 GPU 训练
torchrun --nproc_per_node=4 train_denoise.py --config config/config_denoise_semanticstf --distributed

# 后台训练 (使用 nohup)
nohup python train_denoise.py --config config/config_denoise_semanticstf > train.log 2>&1 &

# 后台训练 (使用 screen)
screen -S train
python train_denoise.py --config config/config_denoise_semanticstf
# Ctrl+A+D 分离会话
# screen -r train 恢复会话
```

---

## 常见问题

### Q: CUDA 扩展编译失败

```bash
# 检查 CUDA 是否安装
nvcc --version

# 如果没有，安装 CUDA toolkit
# Ubuntu
sudo apt-get install nvidia-cuda-toolkit

# 或从 NVIDIA 官网下载安装
```

### Q: 内存不足

```python
# 修改 config/config_denoise_semanticstf.py
frame_point_num = 40000      # 减少采样点数
batch_size_per_gpu = 2       # 减小 batch size
```

### Q: 训练中断后恢复

```bash
# 修改 train_denoise.py，添加 resume 参数
# 或直接修改配置中的 begin_epoch
```
