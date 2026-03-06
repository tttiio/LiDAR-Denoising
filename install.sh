#!/bin/bash
# ============================================================
# LiDAR Point Cloud Denoising - Installation Script
# ============================================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "LiDAR Point Cloud Denoising Installer"
echo "========================================"
echo ""

# 检查 Python 版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# 检查是否有 conda
if command -v conda &> /dev/null; then
    echo "Conda detected: $(conda --version)"
fi

# 检查是否有 CUDA
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo "PyTorch not installed yet"

echo ""
echo "========================================"
echo "Installing dependencies..."
echo "========================================"

# 安装核心依赖
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Building CUDA/CPU extensions..."
echo "========================================"

# 编译扩展
python setup.py build_ext --inplace

echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"

# 验证安装
python -c "
try:
    import torch
    print(f'✓ PyTorch {torch.__version__} installed')
    
    import point_deep.cpu_kernel
    print('✓ CPU kernel compiled')
    
    try:
        import point_deep.cuda_kernel
        print('✓ CUDA kernel compiled')
    except ImportError:
        print('⚠ CUDA kernel not compiled (CPU-only mode)')
    
    from models_denoise import DenoiseNet
    print('✓ DenoiseNet imported')
    
    from utils.self_supervised_loss import SelfSupervisedDenoiseLoss
    print('✓ Loss functions imported')
    
    print()
    print('🎉 Installation successful!')
    
except Exception as e:
    print(f'✗ Installation failed: {e}')
    exit(1)
"

echo ""
echo "========================================"
echo "Next steps:"
echo "========================================"
echo "1. Prepare your data in data/SemanticSTF/"
echo "2. Run training: python train_denoise.py --config config/config_denoise_semanticstf"
echo "3. Run evaluation: python evaluate_denoise.py --config config/config_denoise_semanticstf --checkpoint <model_path>"
echo ""
