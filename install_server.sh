#!/bin/bash
# ============================================================
# LiDAR Point Cloud Denoising - 一键安装脚本
# 适用于阿里云服务器 (Ubuntu/CentOS)
# ============================================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "LiDAR 点云去噪系统 - 自动安装脚本"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# 1. 检查系统环境
# ============================================================
print_info "检查系统环境..."

# 检查操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
    print_info "操作系统: $OS $VER"
else
    print_error "无法检测操作系统"
    exit 1
fi

# 检查是否为 root 用户
if [ "$EUID" -eq 0 ]; then
    print_warn "检测到 root 用户，建议使用普通用户运行"
fi

# 检查 GPU
print_info "检查 GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_info "CUDA 版本: $CUDA_VERSION"
    HAS_GPU=true
else
    print_warn "未检测到 NVIDIA GPU，将安装 CPU 版本"
    HAS_GPU=false
fi

echo ""

# ============================================================
# 2. 安装 Miniconda
# ============================================================
print_info "步骤 1/6: 安装 Miniconda..."

CONDA_PATH="$HOME/miniconda3"

if [ -d "$CONDA_PATH" ]; then
    print_info "Miniconda 已安装在 $CONDA_PATH"
else
    print_info "下载 Miniconda..."
    wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    
    print_info "安装 Miniconda..."
    bash /tmp/miniconda.sh -b -p $CONDA_PATH
    
    print_info "初始化 conda..."
    $CONDA_PATH/bin/conda init bash
    
    # 清理安装包
    rm /tmp/miniconda.sh
    
    print_info "Miniconda 安装完成!"
fi

# 确保 conda 可用
export PATH="$CONDA_PATH/bin:$PATH"
eval "$($CONDA_PATH/bin/conda shell.bash hook)"

conda --version
echo ""

# ============================================================
# 3. 创建 conda 环境
# ============================================================
print_info "步骤 2/6: 创建 conda 环境..."

ENV_NAME="denoise"

if conda env list | grep -q "^$ENV_NAME "; then
    print_info "环境 '$ENV_NAME' 已存在"
else
    print_info "创建环境 '$ENV_NAME' (Python 3.10)..."
    conda create -n $ENV_NAME python=3.10 -y
    print_info "环境创建完成!"
fi

# 激活环境
print_info "激活环境..."
conda activate $ENV_NAME
echo ""

# ============================================================
# 4. 安装 PyTorch
# ============================================================
print_info "步骤 3/6: 安装 PyTorch..."

if [ "$HAS_GPU" = true ]; then
    # 根据 CUDA 版本选择 PyTorch
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        print_info "安装 PyTorch (CUDA 12.1)..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    else
        print_info "安装 PyTorch (CUDA 11.8)..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    fi
else
    print_info "安装 PyTorch (CPU 版本)..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo ""

# ============================================================
# 5. 安装其他依赖
# ============================================================
print_info "步骤 4/6: 安装其他依赖..."

pip install numpy scipy pyyaml matplotlib seaborn scikit-learn tqdm

# 可选: open3d 用于可视化
read -p "是否安装 open3d 用于可视化? (y/n, 默认 y): " install_open3d
install_open3d=${install_open3d:-y}

if [ "$install_open3d" = "y" ] || [ "$install_open3d" = "Y" ]; then
    pip install open3d
fi

echo ""

# ============================================================
# 6. 编译 CUDA 扩展
# ============================================================
print_info "步骤 5/6: 编译 CUDA/CPU 扩展..."

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ "$HAS_GPU" = true ]; then
    print_info "编译 CUDA 扩展..."
    FORCE_CUDA=1 python setup.py build_ext --inplace
else
    print_info "编译 CPU 扩展..."
    python setup.py build_ext --inplace
fi

echo ""

# ============================================================
# 7. 验证安装
# ============================================================
print_info "步骤 6/6: 验证安装..."

python << 'PYEOF'
import sys

print("-" * 50)
print("验证安装...")
print("-" * 50)

errors = []

# 检查 PyTorch
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
    else:
        print("⚠ CUDA 不可用，使用 CPU 模式")
except ImportError as e:
    print(f"✗ PyTorch 导入失败: {e}")
    errors.append("PyTorch")

# 检查 CUDA 扩展
try:
    import point_deep.cuda_kernel
    print("✓ CUDA 扩展编译成功")
except ImportError:
    try:
        import point_deep.cpu_kernel
        print("✓ CPU 扩展编译成功")
    except ImportError as e:
        print(f"✗ 扩展编译失败: {e}")
        print("  将使用 PyTorch fallback 实现")
        errors.append("Extension")

# 检查模型
try:
    from models_denoise import DenoiseNet
    print("✓ DenoiseNet 导入成功")
except ImportError as e:
    print(f"✗ 模型导入失败: {e}")
    errors.append("Model")

# 检查损失函数
try:
    from utils.self_supervised_loss import SelfSupervisedDenoiseLoss
    print("✓ 损失函数导入成功")
except ImportError as e:
    print(f"✗ 损失函数导入失败: {e}")
    errors.append("Loss")

# 检查数据加载器
try:
    from datasets.semanticstf_data import DataloadTrain
    print("✓ 数据加载器导入成功")
except ImportError as e:
    print(f"✗ 数据加载器导入失败: {e}")
    errors.append("DataLoader")

print("-" * 50)

if errors:
    print(f"\n⚠ 安装完成，但以下组件有问题: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n🎉 所有组件安装成功!")
    sys.exit(0)

PYEOF

INSTALL_STATUS=$?

echo ""
echo "========================================"
echo "安装完成!"
echo "========================================"
echo ""

if [ $INSTALL_STATUS -eq 0 ]; then
    echo "下一步操作:"
    echo ""
    echo "1. 激活环境:"
    echo "   source ~/miniconda3/etc/profile.d/conda.sh"
    echo "   conda activate denoise"
    echo ""
    echo "2. 准备数据:"
    echo "   将 SemanticSTF 数据集放到 data/SemanticSTF/ 目录"
    echo ""
    echo "3. 运行测试:"
    echo "   python test_denoise.py"
    echo ""
    echo "4. 开始训练:"
    echo "   python train_denoise.py --config config/config_denoise_semanticstf"
    echo ""
else
    echo "⚠ 安装过程中遇到问题，请检查上面的错误信息"
    echo ""
    echo "常见问题:"
    echo "1. CUDA 扩展编译失败: 确保 NVIDIA 驱动和 CUDA toolkit 已安装"
    echo "2. 内存不足: 使用较小的 batch_size 或 frame_point_num"
    echo ""
fi

# 保存环境信息
print_info "保存环境信息到 env_info.txt..."
python << 'PYEOF'
import torch
import sys

with open("env_info.txt", "w") as f:
    f.write("=" * 50 + "\n")
    f.write("环境信息\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Python: {sys.version}\n")
    f.write(f"PyTorch: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    if torch.cuda.is_available():
        f.write(f"CUDA version: {torch.version.cuda}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    f.write("\n")
PYEOF

print_info "环境信息已保存到 env_info.txt"
echo ""
echo "完成!"
