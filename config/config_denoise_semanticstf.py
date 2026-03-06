"""
SemanticSTF 数据集配置

用于自监督点云去噪训练
"""

import os


def get_config():
    class General:
        log_frequency = 50
        name = "denoise_semanticstf"
        batch_size_per_gpu = 4
        fp16 = True

        # SemanticSTF 数据集路径
        SeqDir = './data/SemanticSTF'

        class Voxel:
            RV_theta = (-40.0, 20.0)
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-5.0, 3.0)

            bev_shape = (600, 600, 30)
            rv_shape = (64, 2048)

    class DatasetParam:
        class Train:
            data_src = 'semanticstf_data'
            num_workers = 4
            frame_point_num = 60000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            
            class AugParam:
                noise_mean = 0
                noise_std = 0.01
                theta_range = (-180.0, 180.0)
                shift_range = ((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2))
                size_range = (0.95, 1.05)

        class Val:
            data_src = 'semanticstf_data'
            num_workers = 4
            frame_point_num = 60000  # 验证时也采样，避免内存溢出
            SeqDir = General.SeqDir
            Voxel = General.Voxel

    class ModelParam:
        prefix = "denoise_net"
        Voxel = General.Voxel
        
        # 特征维度
        xyz_feat_dim = 64
        intensity_feat_dim = 64
        fused_feat_dim = 128

        class BEVParam:
            base_block = 'BasicBlock'
            context_layers = (64, 128, 256, 512)
            layers = (1, 1, 2)
            bev_grid2point = dict(type='BilinearSample', scale_rate=(0.5, 0.5))

        class RVParam:
            base_block = 'BasicBlock'
            context_layers = (64, 128, 256, 512)
            layers = (1, 1, 2)
            rv_grid2point = dict(type='BilinearSample', scale_rate=(1.0, 0.5))

    class LossParam:
        # 自监督损失参数
        mask_ratio = 0.15
        knn_k = 8
        sharpness = 2.0
        sparsity_threshold = 0.1
        temperature = 0.1
        anomaly_threshold = 0.5

        # 邻域重建损失参数（核心）
        neighborhood_k = 16  # 近邻数量
        min_neighbors = 5  # 最少有效邻居数
        max_points = 6000  # 采样点数限制

        # 表面平滑损失参数
        smoothness_k = 8  # 平滑约束近邻数

        # 损失权重
        w_intensity = 1.0
        w_spatial = 0.5
        w_sparsity = 0.3
        w_contrastive = 0.2
        w_dist = 0.5
        w_neighborhood = 1.0  # 邻域重建权重（重要）
        w_smoothness = 0.3  # 表面平滑权重

    class OptimizeParam:
        class optimizer:
            type = "adamw"
            base_lr = 1e-3
            weight_decay = 1e-4
            betas = (0.9, 0.999)

        class schedule:
            type = "cosine"
            begin_epoch = 0
            end_epoch = 50
            warmup_epochs = 5
            min_lr = 1e-6

    return General, DatasetParam, ModelParam, OptimizeParam
