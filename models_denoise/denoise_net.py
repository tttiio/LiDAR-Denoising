"""
自监督点云去噪网络

核心思想：
1. 笛卡尔分支(BEV)：处理 xyz 空间信息，学习几何一致性
2. 极坐标分支(Range View)：处理 intensity 反射强度，学习强度预测
3. 异常检测：通过预测误差判断异常点
4. 位移预测：预测点应该移动到的正确位置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from networks import backbone, bird_view, range_view
from networks.backbone import get_module
import pytorch_lib


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = pytorch_lib.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat


class IntensityPredictor(nn.Module):
    """
    强度预测模块：基于邻域特征预测中心点强度
    正常点预测准确，噪声点预测误差大
    """
    def __init__(self, feat_dim, hidden_dim=64):
        super(IntensityPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_dim=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)  # 输出预测强度
        )
    
    def forward(self, x):
        """
        Input: (BS, C, N, 1)
        Output: (BS, 1, N, 1) 预测的强度值
        """
        return self.predictor(x)


class SpatialConsistencyModule(nn.Module):
    """
    空间一致性模块：学习局部几何的平滑性
    预测每个点与邻域的偏离程度
    """
    def __init__(self, feat_dim):
        super(SpatialConsistencyModule, self).__init__()
        self.consistency_net = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()  # 输出一致性分数 [0, 1]，越大越正常
        )
    
    def forward(self, x):
        """
        Input: (BS, C, N, 1)
        Output: (BS, 1, N, 1) 一致性分数
        """
        return self.consistency_net(x)


class DisplacementPredictor(nn.Module):
    """
    位移预测模块：预测点应该移动的位移向量
    正常点 Δ ≈ 0，噪声点 Δ ≠ 0
    """
    def __init__(self, feat_dim, hidden_dim=64):
        super(DisplacementPredictor, self).__init__()
        self.disp_net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, kernel_size=1)  # 输出 Δx, Δy, Δz
        )
    
    def forward(self, x):
        """
        Input: (BS, C, N, 1)
        Output: (BS, 3, N, 1) 位移向量
        """
        return self.disp_net(x)


class AnomalyDetector(nn.Module):
    """
    异常检测模块：融合强度误差和空间一致性，判断异常点
    """
    def __init__(self, feat_dim):
        super(AnomalyDetector, self).__init__()
        self.detector = nn.Sequential(
            nn.Conv2d(feat_dim + 2, feat_dim // 2, kernel_size=1),  # +2: intensity_error, spatial_score
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=1),
            nn.BatchNorm2d(feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()  # 异常概率 [0, 1]
        )
    
    def forward(self, fused_feat, intensity_error, spatial_score):
        """
        Input:
            fused_feat: (BS, C, N, 1) 融合特征
            intensity_error: (BS, 1, N, 1) 强度预测误差
            spatial_score: (BS, 1, N, 1) 空间一致性分数
        Output:
            anomaly_prob: (BS, 1, N, 1) 异常概率
        """
        # 误差归一化
        intensity_error_norm = intensity_error / (intensity_error.max() + 1e-6)
        
        # 拼接特征
        concat_feat = torch.cat([fused_feat, intensity_error_norm, spatial_score], dim=1)
        
        return self.detector(concat_feat)


class DenoiseNet(nn.Module):
    """
    自监督点云去噪网络
    
    输入：
        - xyz: 笛卡尔坐标
        - intensity: 反射强度
        - 原始点云特征
    
    输出：
        - 异常检测结果
        - 位移预测
        - 去噪后的点云
    """
    def __init__(self, pModel):
        super(DenoiseNet, self).__init__()
        self.pModel = pModel
        
        # 体素参数
        self.bev_shape = list(pModel.Voxel.bev_shape)
        self.rv_shape = list(pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]
        
        self.dx = (pModel.Voxel.range_x[1] - pModel.Voxel.range_x[0]) / pModel.Voxel.bev_shape[0]
        self.dy = (pModel.Voxel.range_y[1] - pModel.Voxel.range_y[0]) / pModel.Voxel.bev_shape[1]
        self.dz = (pModel.Voxel.range_z[1] - pModel.Voxel.range_z[0]) / pModel.Voxel.bev_shape[2]
        
        # 特征维度
        self.xyz_feat_dim = pModel.xyz_feat_dim  # 笛卡尔分支特征维度
        self.intensity_feat_dim = pModel.intensity_feat_dim  # 极坐标分支特征维度
        self.fused_feat_dim = pModel.fused_feat_dim  # 融合后特征维度
        
        self.build_network()
    
    def build_network(self):
        """构建网络各模块"""
        
        # ========== 输入编码 ==========
        # xyz 编码器 (笛卡尔分支)
        self.xyz_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),  # 输入: x, y, z
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.xyz_feat_dim, kernel_size=1),
            nn.BatchNorm2d(self.xyz_feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # intensity 编码器 (极坐标分支)
        self.intensity_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1),  # 输入: intensity
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.intensity_feat_dim, kernel_size=1),
            nn.BatchNorm2d(self.intensity_feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # ========== BEV 分支 (处理 xyz) ==========
        bev_context_layer = self.pModel.BEVParam.context_layers
        bev_layers = self.pModel.BEVParam.layers
        bev_base_block = self.pModel.BEVParam.base_block
        bev_grid2point = self.pModel.BEVParam.bev_grid2point
        
        self.bev_net = bird_view.BEVNet(bev_base_block, bev_context_layer, bev_layers)
        self.bev_grid2point = get_module(bev_grid2point, in_dim=self.bev_net.out_channels[-1])
        
        # ========== Range View 分支 (处理 intensity) ==========
        rv_context_layer = self.pModel.RVParam.context_layers
        rv_layers = self.pModel.RVParam.layers
        rv_base_block = self.pModel.RVParam.base_block
        rv_grid2point = self.pModel.RVParam.rv_grid2point
        
        self.rv_net = range_view.RVNet(rv_base_block, rv_context_layer, rv_layers)
        self.rv_grid2point = get_module(rv_grid2point, in_dim=self.rv_net.out_channels[-1])
        
        # ========== 特征融合 ==========
        fusion_in_dim = self.bev_net.out_channels[-1] + self.rv_net.out_channels[-1]
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_dim, self.fused_feat_dim, kernel_size=1),
            nn.BatchNorm2d(self.fused_feat_dim),
            nn.ReLU(inplace=True)
        )
        
        # ========== 自监督任务头 ==========
        # 强度预测（极坐标分支）
        self.intensity_predictor = IntensityPredictor(self.rv_net.out_channels[-1])
        
        # 空间一致性（笛卡尔分支）
        self.spatial_consistency = SpatialConsistencyModule(self.bev_net.out_channels[-1])
        
        # 异常检测
        self.anomaly_detector = AnomalyDetector(self.fused_feat_dim)
        
        # 位移预测
        self.displacement_predictor = DisplacementPredictor(self.fused_feat_dim)
        
        # ========== 辅助模块 ==========
        # 点特征预处理的 PointNet
        self.point_pre_xyz = backbone.PointNetStacker(self.xyz_feat_dim, bev_context_layer[0], pre_bn=False, stack_num=1)
        self.point_pre_intensity = backbone.PointNetStacker(self.intensity_feat_dim, rv_context_layer[0], pre_bn=False, stack_num=1)
    
    def forward(self, pcds_xyz, pcds_intensity, pcds_coord, pcds_sphere_coord):
        """
        前向传播
        
        Input:
            pcds_xyz: (BS, 3, N, 1) 笛卡尔坐标 x, y, z
            pcds_intensity: (BS, 1, N, 1) 反射强度
            pcds_coord: (BS, N, 2, 1) BEV 坐标
            pcds_sphere_coord: (BS, N, 2, 1) 极坐标
        
        Output:
            pred_dict: 包含各种预测结果
        """
        BS = pcds_xyz.shape[0]
        N = pcds_xyz.shape[2]
        
        # ========== 1. 编码输入 ==========
        xyz_feat = self.xyz_encoder(pcds_xyz)  # (BS, C1, N, 1)
        intensity_feat = self.intensity_encoder(pcds_intensity)  # (BS, C2, N, 1)
        
        # ========== 2. 笛卡尔分支 (BEV) 处理 xyz ==========
        xyz_feat_pre = self.point_pre_xyz(xyz_feat)
        bev_input = VoxelMaxPool(xyz_feat_pre, pcds_coord, output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0))
        bev_feat_past, bev_feat = self.bev_net(bev_input)
        point_bev_feat = self.bev_grid2point(bev_feat, pcds_coord)  # (BS, C, N, 1)
        
        # ========== 3. 极坐标分支 (Range View) 处理 intensity ==========
        intensity_feat_pre = self.point_pre_intensity(intensity_feat)
        rv_input = VoxelMaxPool(intensity_feat_pre, pcds_sphere_coord, output_size=self.rv_shape, scale_rate=(1.0, 1.0))
        rv_feat_past, rv_feat = self.rv_net(rv_input)
        point_rv_feat = self.rv_grid2point(rv_feat, pcds_sphere_coord)  # (BS, C, N, 1)
        
        # ========== 4. 自监督任务 ==========
        # 4.1 强度预测
        pred_intensity = self.intensity_predictor(point_rv_feat)  # (BS, 1, N, 1)
        
        # 4.2 空间一致性
        spatial_score = self.spatial_consistency(point_bev_feat)  # (BS, 1, N, 1)
        
        # ========== 5. 特征融合 ==========
        fused_feat = torch.cat([point_bev_feat, point_rv_feat], dim=1)
        fused_feat = self.fusion(fused_feat)  # (BS, C, N, 1)
        
        # ========== 6. 强度预测误差 ==========
        intensity_error = (pred_intensity - pcds_intensity).abs()  # (BS, 1, N, 1)
        
        # ========== 7. 异常检测 ==========
        anomaly_prob = self.anomaly_detector(fused_feat, intensity_error, spatial_score)  # (BS, 1, N, 1)
        
        # ========== 8. 位移预测 ==========
        displacement = self.displacement_predictor(fused_feat)  # (BS, 3, N, 1)
        
        # ========== 9. 去噪后的点云 ==========
        denoised_xyz = pcds_xyz + displacement * anomaly_prob  # 只有异常点才移动
        
        # 返回所有预测结果
        pred_dict = {
            'pred_intensity': pred_intensity,
            'spatial_score': spatial_score,
            'intensity_error': intensity_error,
            'anomaly_prob': anomaly_prob,
            'displacement': displacement,
            'denoised_xyz': denoised_xyz,
            # 中间特征（用于损失计算）
            'point_bev_feat': point_bev_feat,
            'point_rv_feat': point_rv_feat,
            'fused_feat': fused_feat
        }
        
        return pred_dict
    
    def infer(self, pcds_xyz, pcds_intensity, pcds_coord, pcds_sphere_coord):
        """
        推理模式：只返回去噪结果
        """
        self.eval()
        with torch.no_grad():
            pred_dict = self.forward(pcds_xyz, pcds_intensity, pcds_coord, pcds_sphere_coord)
        
        return pred_dict['denoised_xyz'], pred_dict['anomaly_prob']
