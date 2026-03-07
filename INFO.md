# 自监督点云去噪系统

## 一、研究问题

### 1.1 目标
解决雨雾天气下LiDAR点云噪声问题（雨滴、雪花、雾气造成的虚假点）

### 1.2 挑战
- 真实场景缺乏噪声标注
- 传统方法依赖手工阈值，泛化性差
- 需要同时检测噪声并恢复正确位置

---

## 二、核心思想

### 2.1 自监督学习
不需要标注，利用数据内在结构学习

### 2.2 关键假设
| 点类型 | 特征 |
|--------|------|
| **正常点** | 局部几何平滑、强度预测准确 |
| **噪声点** | 偏离邻域、强度预测误差大 |

---

## 三、网络架构

### 3.1 双分支结构

```
输入点云 (xyz, intensity)
    │
    ├── 笛卡尔分支 (BEV)
    │   ├─ 输入: xyz 空间坐标
    │   ├─ 处理: BEV特征提取
    │   └─ 输出: 空间一致性分数 (0-1)
    │
    └── 极坐标分支 (Range View)
        ├─ 输入: intensity 反射强度
        ├─ 处理: Range View特征提取
        └─ 输出: 强度预测值
    │
    ↓
特征融合 (128维)
    │
    ├──→ 异常检测模块 → 噪声概率 (0-1)
    │
    └──→ 位移预测模块 → 位移向量 (Δx, Δy, Δz)
    │
    ↓
去噪结果 = 原始点云 + 位移 × 异常概率
```

### 3.2 自监督任务

| 任务 | 输入 | 输出 | 正常点行为 | 噪声点行为 |
|------|------|------|------------|------------|
| **强度预测** | 邻域特征 | 预测强度 | 预测准确 | 预测误差大 |
| **空间一致性** | 局部几何 | 一致性分数 | 高(接近邻域) | 低(偏离邻域) |
| **位移预测** | 融合特征 | 位移向量 | ≈ 0 | ≠ 0 |
| **异常检测** | 误差+一致性 | 噪声概率 | 低(< 0.5) | 高(> 0.5) |

---

## 四、损失函数设计

### 4.1 总损失公式

```
L_total = λ₁·L_intensity      (强度预测损失)
        + λ₂·L_spatial        (空间一致性损失)
        + λ₃·L_sparsity       (位移稀疏损失)
        + λ₄·L_contrastive    (对比学习损失)
        + λ₅·L_dist           (强度分布损失)
        + λ₆·L_neighborhood   (邻域重建损失 - 核心创新)
        + λ₇·L_smoothness     (表面平滑损失)
```

### 4.2 各损失函数详解

#### ① 强度预测损失 (L_intensity)
- **思路**: 随机mask 15%点的强度，用邻域特征预测
- **原理**: 正常点预测准确，噪声点预测误差大
- **权重**: 1.0

#### ② 空间一致性损失 (L_spatial)
- **思路**: 正常点云局部几何平滑连续
- **原理**: 计算点到K近邻中心的距离，距离大则可能是噪声
- **权重**: 0.5

#### ③ 位移稀疏损失 (L_sparsity)
- **思路**: 大部分点是正常的，位移应接近0
- **原理**: 只有少数异常点需要大位移，通过L1稀疏约束
- **权重**: 0.3

#### ④ 对比学习损失 (L_contrastive)
- **思路**: 正常点之间特征相似，正常点与异常点特征不同
- **原理**: 在特征空间增强判别能力
- **权重**: 0.2

#### ⑤ 强度分布损失 (L_dist)
- **思路**: 雨雾噪声点强度有特定分布模式
- **原理**: 用空间一致性分数加权，高一致性点的强度预测应更准
- **权重**: 0.5

#### ⑥ 邻域重建损失 (L_neighborhood) ⭐ 核心创新
- **思路**: 异常点应移动到邻域正常点拟合的平面上
- **实现步骤**:
  1. 找K近邻 (k=16)
  2. 用正常点（低异常概率）加权拟合局部平面（PCA）
  3. 计算去噪后的点到平面的距离
  4. 高异常概率的点，距离应该小
- **效果**: 提供几何约束，引导噪声点移动到正确位置
- **权重**: 1.0

#### ⑦ 表面平滑损失 (L_smoothness)
- **思路**: 去噪后表面应平滑，相邻点位移应相似
- **权重**: 0.3

---

## 五、训练配置

### 5.1 数据配置

| 配置项 | 值 |
|--------|-----|
| 数据集 | SemanticSTF |
| 训练样本 | 1326帧 |
| 验证样本 | 250帧 |
| 点数/帧 | 60000 (采样) |
| 噪声标签 | unlabeled=0, invalid=20 |

### 5.2 天气分布

| 天气类型 | 训练集 | 验证集 |
|----------|--------|--------|
| rain | 66 | 16 |
| snow | 460 | 78 |
| light_fog | 397 | 78 |
| dense_fog | 403 | 78 |

### 5.3 体素参数

| 参数 | 值 |
|------|-----|
| X范围 | [-50, 50] m |
| Y范围 | [-50, 50] m |
| Z范围 | [-5, 3] m |
| BEV网格 | 600 × 600 × 30 |
| Range View网格 | 64 × 2048 |

### 5.4 模型参数

| 参数 | 值 |
|------|-----|
| xyz特征维度 | 64 |
| intensity特征维度 | 64 |
| 融合特征维度 | 128 |
| BEV layers | (1, 1, 2) |
| RV layers | (1, 1, 2) |
| context_layers | (64, 128, 256, 512) |

### 5.5 损失函数参数

| 参数 | 值 | 说明 |
|------|-----|------|
| mask_ratio | 0.15 | 强度预测mask比例 |
| knn_k | 8 | 空间一致性K近邻数 |
| sharpness | 2.0 | 一致性锐度 |
| sparsity_threshold | 0.1 | 位移稀疏阈值 |
| temperature | 0.1 | 对比学习温度 |
| anomaly_threshold | 0.5 | 异常判定阈值 |
| neighborhood_k | 16 | 邻域重建K近邻数 |
| min_neighbors | 5 | 最少有效邻居数 |
| max_points | 6000 | 采样点数限制 |
| smoothness_k | 8 | 表面平滑K近邻数 |

### 5.6 损失权重

| 损失项 | 权重 |
|--------|------|
| w_intensity | 1.0 |
| w_spatial | 0.5 |
| w_sparsity | 0.3 |
| w_contrastive | 0.2 |
| w_dist | 0.5 |
| w_neighborhood | 1.0 |
| w_smoothness | 0.3 |

### 5.7 优化器配置

| 配置项 | 值 |
|--------|-----|
| 优化器 | AdamW |
| 初始学习率 | 1e-3 |
| 最终学习率 | 1e-6 |
| weight_decay | 1e-4 |
| betas | (0.9, 0.999) |

### 5.8 学习率调度

| 配置项 | 值 |
|--------|-----|
| 调度器 | OneCycle |
| 总轮数 | 50 epochs |
| warmup比例 | 10% |
| 验证频率 | 每5个epoch |

### 5.9 训练配置

| 配置项 | 值 |
|--------|-----|
| Batch Size | 4 |
| 混合精度 | FP16 |
| 日志频率 | 每50 iter |
| num_workers | 4 |

---

## 六、训练关键优化

### 6.1 AMP 防治机制
```python
# 混合精度训练时，如果遇到 NaN/Inf，优化器会跳过参数更新
# 此时不应更新学习率，否则会导致学习率调度混乱
scale = scaler.get_scale()
scaler.update()
skip_lr_sched = (scale > scaler.get_scale())
if not skip_lr_sched:
    scheduler.step()
```

### 6.2 内存优化策略
- 各损失函数中使用随机采样（max_points=6000~8000）
- 避免KNN计算时内存溢出

---

## 七、评估方案

### 7.1 评估命令
```bash
python evaluate_denoise.py \
    --config config/config_denoise_semanticstf \
    --checkpoint experiments/denoise_semanticstf/checkpoint/best_model.pth \
    --output eval_results
```

### 7.2 评估指标

| 指标 | 含义 | 目标值 |
|------|------|--------|
| Accuracy | 整体准确率 | > 0.90 |
| Precision | 预测为噪声中真噪声比例 | > 0.85 |
| Recall | 真实噪声被正确检测比例 | > 0.80 |
| F1 Score | Precision和Recall调和平均 | > 0.82 |
| ROC-AUC | ROC曲线下面积 | > 0.90 |
| PR-AUC | PR曲线下面积 | > 0.85 |

### 7.3 可视化输出

评估脚本会在 `eval_results/` 目录生成：
- `roc_curve.png` - ROC曲线
- `pr_curve.png` - PR曲线
- `confusion_matrix.png` - 混淆矩阵
- `probability_distribution.png` - 异常概率分布
- `evaluation_report.txt` - 详细评估报告（含各天气类型性能）

---

## 八、文件结构

```
LiDAR-Denoising/
├── config/
│   └── config_denoise_semanticstf.py   # 配置文件
├── models_denoise/
│   ├── __init__.py
│   └── denoise_net.py                  # 去噪网络
├── networks/
│   ├── backbone.py                     # 骨干网络
│   ├── bird_view.py                    # BEV分支
│   └── range_view.py                   # Range View分支
├── datasets/
│   └── semanticstf_data.py             # 数据加载
├── utils/
│   ├── self_supervised_loss.py         # 自监督损失函数
│   ├── builder.py                      # 优化器构建
│   └── logger.py                       # 日志工具
├── train_denoise.py                    # 训练脚本
├── evaluate_denoise.py                 # 评估脚本
├── inference_denoise.py                # 推理脚本
├── test_denoise.py                     # 测试脚本
├── INFO.md                             # 本文件
└── README.md                           # 项目说明
```

---

## 九、运行命令

### 9.1 训练
```bash
# 单GPU训练
python train_denoise.py --config config/config_denoise_semanticstf

# 分布式训练
torchrun --nproc_per_node=N train_denoise.py \
    --config config/config_denoise_semanticstf --distributed
```

### 9.2 评估
```bash
python evaluate_denoise.py \
    --config config/config_denoise_semanticstf \
    --checkpoint experiments/denoise_semanticstf/checkpoint/best_model.pth \
    --output eval_results
```

### 9.3 推理
```bash
python inference_denoise.py \
    --config config/config_denoise_semanticstf \
    --checkpoint experiments/denoise_semanticstf/checkpoint/best_model.pth \
    --input your_pointcloud.bin \
    --output denoised_pointcloud.bin
```

---

## 十、预期结果

- 整体 F1 Score > 0.80
- ROC-AUC > 0.90
- 能有效检测雨滴、雪花、雾气噪声
- 去噪后点云保留真实物体结构

---

## 十一、下一步计划

1. 等待训练完成
2. 运行评估脚本验证性能
3. 可视化去噪效果（对比去噪前后）
4. 消融实验（验证各损失函数贡献）
5. 跨数据集测试泛化性
