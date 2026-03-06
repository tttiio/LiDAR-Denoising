# LiDAR Point Cloud Denoising

自监督点云去噪系统，用于雨雾天气下的 LiDAR 点云数据处理。

## 🌟 特性

- **双分支架构**: 笛卡尔分支(BEV)处理空间几何，极坐标分支(RV)处理反射强度
- **自监督学习**: 无需标注数据，通过多个自监督任务学习去噪
- **异常检测**: 自动识别雨雾噪声点
- **点云恢复**: 将噪声点恢复到正确位置

## 📋 环境要求

- Python >= 3.7
- PyTorch >= 1.9.0
- CUDA >= 10.2 (GPU 训练)

## 🚀 安装

### 方式一: 使用 pip 安装依赖

```bash
# 创建 conda 环境
conda create -n denoise python=3.10
conda activate denoise

# 安装 PyTorch (GPU 版本)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt

# 编译 CUDA 扩展
python setup.py build_ext --inplace
```

### 方式二: 开发模式安装

```bash
# 完整安装（包含可视化工具）
pip install -e ".[visualize]"

# 或者仅核心依赖
pip install -e .
```

### 强制编译 CUDA 扩展

```bash
# 如果自动检测失败，强制编译 CUDA
FORCE_CUDA=1 python setup.py build_ext --inplace
```

## 📁 项目结构

```
LiDAR-Denoising/
├── config/                     # 配置文件
│   └── config_denoise_semanticstf.py
├── data/                       # 数据目录
│   └── SemanticSTF/
│       ├── train/
│       ├── val/
│       └── test/
├── datasets/                   # 数据加载器
│   ├── semanticstf_data.py    # SemanticSTF 数据集
│   ├── denoise_data.py        # 通用去噪数据集
│   └── utils.py               # 数据处理工具
├── models_denoise/            # 去噪模型
│   └── denoise_net.py         # 主网络
├── networks/                  # 网络模块
│   ├── backbone.py            # 基础网络
│   ├── bird_view.py           # BEV 分支
│   └── range_view.py          # RV 分支
├── utils/                     # 工具函数
│   ├── self_supervised_loss.py # 损失函数
│   ├── builder.py             # 优化器构建
│   └── logger.py              # 日志工具
├── pytorch_lib/               # CUDA/CPU 扩展
├── train_denoise.py           # 训练脚本
├── evaluate_denoise.py        # 评估脚本
├── inference_denoise.py       # 推理脚本
├── requirements.txt           # 依赖列表
└── setup.py                   # 安装脚本
```

## 🎯 使用方法

### 训练

```bash
# 单 GPU 训练
python train_denoise.py --config config/config_denoise_semanticstf

# 多 GPU 分布式训练
torchrun --nproc_per_node=4 train_denoise.py \
    --config config/config_denoise_semanticstf \
    --distributed
```

### 评估

```bash
python evaluate_denoise.py \
    --config config/config_denoise_semanticstf \
    --checkpoint experiments/denoise_semanticstf/checkpoint/best_model.pth \
    --output eval_results
```

### 推理

```bash
python inference_denoise.py \
    --config config/config_denoise_semanticstf \
    --checkpoint experiments/denoise_semanticstf/checkpoint/best_model.pth \
    --input your_point_cloud.bin \
    --output results/ \
    --visualize
```

## 📊 模型架构

```
输入点云 (N, 5)
    │
    ├── BEV 分支 ───────────────────────────── RV 分支
    │   xyz 编码                                intensity 编码
    │   VoxelMaxPool → BEVNet                   VoxelMaxPool → RVNet
    │   BilinearSample                          BilinearSample
    │       │                                       │
    │       ├──► 空间一致性 ◄────────────────────────┤
    │       │                                       │
    │       └──────► 特征融合 ◄──────────────────────┘
    │                    │
    │                    ├── 异常检测 → anomaly_prob
    │                    ├── 位移预测 → displacement
    │                    │
    │                    ▼
    │            去噪点云 = xyz + displacement * anomaly_prob
    │
    └── 自监督损失
        ├── 强度预测损失
        ├── 空间一致性损失
        ├── 位移稀疏损失
        ├── 邻域重建损失 ⭐
        └── 表面平滑损失 ⭐
```

## 🔧 配置说明

主要配置在 `config/config_denoise_semanticstf.py`:

```python
# 体素参数
bev_shape = (600, 600, 30)  # BEV 网格大小
rv_shape = (64, 2048)       # Range View 网格大小

# 特征维度
xyz_feat_dim = 64           # 空间特征维度
intensity_feat_dim = 64     # 强度特征维度
fused_feat_dim = 128        # 融合特征维度

# 损失权重
w_intensity = 1.0           # 强度预测
w_neighborhood = 1.0        # 邻域重建 ⭐
w_smoothness = 0.3          # 表面平滑 ⭐
```

## 📈 性能指标

评估指标包括:
- **Precision**: 预测噪声点的准确率
- **Recall**: 噪声点检出率
- **F1 Score**: 综合指标
- **ROC AUC**: 异常检测能力

## 📝 引用

如果这个项目对你有帮助，请引用:

```bibtex
@misc{lidar_denoising,
  title={Self-Supervised LiDAR Point Cloud Denoising for Rain and Fog Weather},
  year={2024}
}
```

## 📄 License

MIT License
