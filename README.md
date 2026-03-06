# LiDAR Point Cloud Denoising

自监督点云去噪系统，用于雨雾天气下的 LiDAR 点云数据处理。

## 🎯 项目目标

本系统解决两个核心问题：

1. **异常点检测**: 判断一个点是否受到雨雾影响的异常点
2. **位置恢复**: 如果是异常点，将其恢复到正确位置

## 🌟 核心特性

- **双分支架构**: 笛卡尔分支(BEV)处理空间几何，极坐标分支(RV)处理反射强度
- **自监督学习**: 无需标注的正确位置，通过邻域重建损失学习恢复
- **异常检测**: 自动识别雨、雪、雾天气产生的噪声点
- **点云恢复**: 将噪声点恢复到邻域拟合的表面

## 📋 环境要求

| 依赖 | 版本 |
|------|------|
| Python | >= 3.7 |
| PyTorch | >= 1.9.0 |
| CUDA | >= 10.2 (GPU 训练) |

## 🚀 安装

### 方式一: 使用 pip

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

### 方式二: 一键安装

```bash
chmod +x install.sh
./install.sh
```

### 验证安装

```bash
python -c "
import torch
import point_deep.cpu_kernel
from models_denoise import DenoiseNet
from utils.self_supervised_loss import SelfSupervisedDenoiseLoss
print('✅ 安装成功!')
"
```

## 📁 项目结构

```
LiDAR-Denoising/
├── config/                         # 配置文件
│   ├── config_denoise_semanticstf.py
│   └── config_denoise_net.py
├── data/                           # 数据目录
│   └── SemanticSTF/
│       ├── train/velodyne/         # 训练点云
│       ├── val/velodyne/           # 验证点云
│       └── test/velodyne/          # 测试点云
├── datasets/                       # 数据加载器
│   ├── semanticstf_data.py        # SemanticSTF 数据集
│   ├── denoise_data.py            # 通用去噪数据集
│   └── utils.py                   # 数据处理工具
├── models_denoise/                # 去噪模型
│   └── denoise_net.py             # 主网络
├── networks/                      # 网络模块
│   ├── backbone.py                # 基础网络组件
│   ├── bird_view.py               # BEV 分支
│   └── range_view.py              # RV 分支
├── utils/                         # 工具函数
│   ├── self_supervised_loss.py   # 损失函数
│   ├── builder.py                 # 优化器构建
│   └── logger.py                  # 日志工具
├── pytorch_lib/                   # CUDA/CPU 扩展
│   ├── src/                       # C++/CUDA 源码
│   └── __init__.py
├── train_denoise.py               # 训练脚本
├── evaluate_denoise.py            # 评估脚本
├── inference_denoise.py           # 推理脚本
├── requirements.txt               # 依赖列表
├── setup.py                       # 安装脚本
└── README.md                      # 本文档
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

评估输出包括：
- ROC 曲线和 AUC
- PR 曲线和 AUC
- 混淆矩阵
- 按天气类型分类的指标

### 推理

```bash
python inference_denoise.py \
    --config config/config_denoise_semanticstf \
    --checkpoint experiments/denoise_semanticstf/checkpoint/best_model.pth \
    --input your_point_cloud.bin \
    --output results/ \
    --threshold 0.5 \
    --visualize
```

## 📊 模型架构

```
输入点云 (N, 5): [x, y, z, intensity, ring]
    │
    ├── BEV 分支 (笛卡尔坐标) ─────────────── RV 分支 (反射强度)
    │   │                                       │
    │   xyz_encoder                             intensity_encoder
    │   │                                       │
    │   VoxelMaxPool → (600, 600)              VoxelMaxPool → (64, 2048)
    │   │                                       │
    │   BEVNet (Encoder-Decoder)               RVNet (Encoder-Decoder)
    │   │                                       │
    │   BilinearSample → point_bev_feat        BilinearSample → point_rv_feat
    │   │                                       │
    │   └───────┬───────────────────────────────────┘
    │           │
    │           ▼
    │     特征融合 (concat + fusion)
    │           │
    │           ├──► 强度预测 → pred_intensity
    │           ├──► 空间一致性 → spatial_score
    │           ├──► 异常检测 → anomaly_prob
    │           └──► 位移预测 → displacement
    │                    │
    │                    ▼
    │           去噪输出: denoised_xyz = xyz + displacement * anomaly_prob
    │
    └── 自监督损失
        ├── MaskedIntensityLoss      (强度预测监督)
        ├── SpatialConsistencyLoss   (空间一致性监督)
        ├── ContrastiveLoss          (特征对比监督)
        ├── NeighborhoodReconstructionLoss ⭐ (位移方向监督)
        ├── DisplacementSparsityLoss (位移稀疏约束)
        └── SurfaceSmoothnessLoss    (表面平滑约束)
```

## 🔬 核心算法

### 1. 异常点检测

通过两个自监督任务识别异常点：

```
异常概率 = f(强度预测误差, 空间一致性分数, 融合特征)

- 强度预测: 正常点预测准确，噪声点预测误差大
- 空间一致性: 正常点离邻域近，噪声点离邻域远
```

### 2. 位置恢复 (邻域重建损失)

```
算法流程:
1. 找到每个异常点的 K 近邻
2. 用邻域中正常点拟合局部平面 (PCA)
3. 约束异常点去噪后位置落在邻域平面上

损失函数:
L_neighborhood = Σ anomaly_prob × distance_to_neighbor_plane
```

## ⚙️ 配置说明

主要配置在 `config/config_denoise_semanticstf.py`:

```python
# 体素参数
bev_shape = (600, 600, 30)    # BEV 网格大小
rv_shape = (64, 2048)          # Range View 网格大小
frame_point_num = 60000        # 每帧采样点数

# 特征维度
xyz_feat_dim = 64              # 空间特征维度
intensity_feat_dim = 64        # 强度特征维度
fused_feat_dim = 128           # 融合特征维度

# 损失权重
w_intensity = 1.0              # 强度预测
w_spatial = 0.5                # 空间一致性
w_neighborhood = 1.0           # 邻域重建 ⭐
w_smoothness = 0.3             # 表面平滑
```

## 📈 评估指标

| 指标 | 说明 |
|------|------|
| Precision | 预测噪声点的准确率 |
| Recall | 噪声点检出率 |
| F1 Score | Precision 和 Recall 的调和平均 |
| ROC AUC | 异常检测能力 |
| PR AUC | 精确率-召回率曲线下面积 |

支持按天气类型分别评估：
- Rain (雨)
- Snow (雪)
- Light Fog (轻雾)
- Dense Fog (浓雾)

## 📝 数据集格式

### SemanticSTF 数据集

```
data/SemanticSTF/
├── train/
│   ├── velodyne/
│   │   ├── 000000.bin    # (N, 5) [x, y, z, intensity, ring]
│   │   └── ...
│   └── labels/
│       ├── 000000.label  # (N,) 语义标签
│       └── ...
├── val/
└── test/
```

### 点云格式

- `.bin` 文件: `np.float32`, 形状 `(N, 5)`
- 列: `[x, y, z, intensity, ring]`
- 标签: `unlabeled=0`, `invalid=20` 为噪声点

## 🔧 故障排除

### CUDA 扩展编译失败

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 强制编译 CUDA
FORCE_CUDA=1 python setup.py build_ext --inplace
```

### 内存溢出 (OOM)

调整配置减少内存使用：

```python
frame_point_num = 40000      # 减少采样点数
max_points = 4000            # 减少损失计算采样
batch_size_per_gpu = 2       # 减小 batch size
```

### M1 Mac 无法运行

M1 Mac 的 MPS 不支持 CUDA 扩展和部分算子，需要在 NVIDIA GPU 上训练。

## 📄 License

MIT License

## 🙏 致谢

- SemanticSTF 数据集
- PyTorch 团队
