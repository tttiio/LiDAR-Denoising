# 自监督点云去噪系统

## 概述

本项目实现了一个自监督点云去噪网络，利用双分支架构处理点云的空间信息和反射强度信息，用于检测和修正雨雾天气下的异常点云。

## 核心架构

```
输入点云 (x, y, z, intensity)
           ↓
┌─────────────────────────────────────────────┐
│  笛卡尔分支(BEV)    极坐标分支(Range View)   │
│  处理 xyz 空间      处理 intensity 强度      │
│         ↓                    ↓              │
│  空间一致性模块      强度预测模块            │
│         ↓                    ↓              │
│              特征融合                        │
│                  ↓                          │
│         异常检测 + 位移预测                  │
│                  ↓                          │
│           去噪后点云                         │
└─────────────────────────────────────────────┘
```

## 文件结构

```
LiDAR-Denoising/
├── models_denoise/
│   ├── __init__.py
│   └── denoise_net.py          # 核心网络架构
├── datasets/
│   ├── semanticstf_data.py     # SemanticSTF 数据加载器
│   └── denoise_data.py         # 通用数据加载器（含噪声模拟）
├── utils/
│   └── self_supervised_loss.py # 自监督损失函数
├── config/
│   ├── config_denoise_semanticstf.py  # SemanticSTF 配置
│   └── config_denoise_net.py          # 通用配置
├── train_denoise.py            # 训练脚本
├── evaluate_denoise.py         # 评估脚本
├── inference_denoise.py        # 推理脚本
└── test_denoise.py             # 测试脚本
```

## 环境配置

### 依赖

```bash
# PyTorch (根据你的CUDA版本选择)
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
pip install numpy scipy pyyaml matplotlib seaborn scikit-learn
pip install open3d nuscenes-devkit tqdm

# 编译 C++ 扩展（VoxelMaxPool）
cd LiDAR-Denoising
python setup.py install
```

### 创建 Conda 环境

```bash
conda create -n denoise python=3.9
conda activate denoise
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pyyaml matplotlib seaborn scikit-learn open3d tqdm
pip install nuscenes-devkit

# 编译扩展
cd LiDAR-Denoising
python setup.py install
```

## 数据集

### SemanticSTF

SemanticSTF 数据集包含真实的雨雾雪天气点云数据：

```
data/SemanticSTF/
├── train/
│   ├── velodyne/      # 点云文件 (.bin)
│   ├── labels/        # 标签文件 (.label)
│   └── train.txt      # 文件列表（含天气类型）
├── val/
│   └── ...
├── test/
│   └── ...
└── semanticstf.yaml   # 标签配置
```

天气类型：
- rain (66 帧)
- snow (460 帧)
- light_fog (397 帧)
- dense_fog (403 帧)

## 使用方法

### 1. 测试代码

```bash
python test_denoise.py
```

### 2. 训练

```bash
# 单 GPU（默认模式）
python train_denoise.py --config config/config_denoise_semanticstf

# 多 GPU 分布式训练（4卡）
torchrun --nproc_per_node=4 train_denoise.py \
    --config config/config_denoise_semanticstf \
    --distributed
```

### 3. 评估

```bash
python evaluate_denoise.py \
    --config config/config_denoise_semanticstf.py \
    --checkpoint experiments/denoise_semanticstf/checkpoint/best_model.pth \
    --output eval_results
```

### 4. 推理

```bash
python inference_denoise.py \
    --config config/config_denoise_semanticstf.py \
    --checkpoint experiments/denoise_semanticstf/checkpoint/best_model.pth \
    --input data/SemanticSTF/test/velodyne/xxx.bin \
    --visualize
```

## 自监督学习原理

网络通过以下自监督任务学习：

| 任务 | 正常点行为 | 噪声点行为 | 用途 |
|------|-----------|-----------|------|
| **强度预测** | 预测准确 | 预测误差大 | 检测强度异常 |
| **空间一致性** | 局部平滑 | 空间突变 | 检测位置异常 |
| **位移稀疏** | Δ ≈ 0 | Δ ≠ 0 | 学习修正向量 |
| **对比学习** | 特征相似 | 特征独特 | 增强判别能力 |

**核心思想**：噪声点在自监督任务中会表现出异常，通过这些异常信号检测并修正。

## 输出说明

### 训练输出

- `experiments/denoise_semanticstf/`
  - `checkpoint/` - 模型检查点
  - `log.txt` - 训练日志

### 评估输出

- `eval_results/`
  - `roc_curve.png` - ROC 曲线
  - `pr_curve.png` - PR 曲线
  - `confusion_matrix.png` - 混淆矩阵
  - `probability_distribution.png` - 概率分布
  - `evaluation_report.txt` - 详细报告

### 推理输出

- 去噪后的点云 (.pcd)
- 检测到的异常点 (.pcd)
- 可视化结果 (.png)

## 超参数调优

主要可调参数（在配置文件中）：

```python
# 损失权重
w_intensity = 1.0    # 强度预测损失
w_spatial = 0.5      # 空间一致性损失
w_sparsity = 0.3     # 位移稀疏损失
w_contrastive = 0.2  # 对比学习损失
w_dist = 0.5         # 强度分布损失

# 自监督参数
mask_ratio = 0.15    # 强度预测的 mask 比例
knn_k = 8            # KNN 邻居数
anomaly_threshold = 0.5  # 异常检测阈值

# 网络参数
xyz_feat_dim = 64           # xyz 特征维度
intensity_feat_dim = 64     # 强度特征维度
fused_feat_dim = 128        # 融合特征维度
```

## 注意事项

1. **VoxelMaxPool** 是 C++ 扩展，需要先编译（`python setup.py install`）
2. 训练时建议使用 FP16 加速（设置 `fp16 = True`）
3. 显存不足时可以减小 `batch_size_per_gpu` 或 `frame_point_num`
4. SemanticSTF 的标签中，`unlabeled=0` 和 `invalid=20` 被视为噪声点

## 下一步改进方向

1. **网络架构**：引入 Transformer 或稀疏卷积
2. **损失函数**：添加边界感知损失
3. **数据增强**：更真实的噪声模拟
4. **多任务学习**：同时做语义分割和去噪
5. **域适应**：适应不同天气条件
