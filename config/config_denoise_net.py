"""
自监督点云去噪网络配置
"""

import os


def get_config():
    class General:
        log_frequency = 50
        name = "denoise_net"
        batch_size_per_gpu = 4
        fp16 = True

        SeqDir = './data/nuscenes'

        class Voxel:
            RV_theta = (-40.0, 20.0)
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-5.0, 3.0)

            bev_shape = (600, 600, 30)
            rv_shape = (64, 2048)

    class DatasetParam:
        class Train:
            data_src = 'denoise_data'
            num_workers = 4
            frame_point_num = 60000
            SeqDir = General.SeqDir
            fname_pkl = os.path.join(SeqDir, 'nuscenes_infos_train.pkl')
            Voxel = General.Voxel
            
            class AugParam:
                noise_mean = 0
                noise_std = 0.01
                theta_range = (-180.0, 180.0)
                shift_range = ((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2))
                size_range = (0.95, 1.05)

        class Val:
            data_src = 'denoise_data'
            num_workers = 4
            frame_point_num = None
            SeqDir = General.SeqDir
            fname_pkl = os.path.join(SeqDir, 'nuscenes_infos_val.pkl')
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

        # 损失权重
        w_intensity = 1.0
        w_spatial = 0.5
        w_sparsity = 0.3
        w_contrastive = 0.2
        w_dist = 0.5

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
