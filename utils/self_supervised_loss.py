"""
自监督损失函数

包含：
1. 强度预测损失：Masked Intensity Prediction
2. 空间一致性损失：局部几何平滑约束
3. 位移稀疏损失：正常点位移应接近0
4. 对比学习损失：正常点 vs 潜在异常点
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedIntensityLoss(nn.Module):
    """
    掩码强度预测损失
    
    思路：
    - 随机mask一部分点的强度
    - 用邻域特征预测被mask的强度
    - 正常点预测准确，噪声点预测误差大
    """
    def __init__(self, mask_ratio=0.15):
        super(MaskedIntensityLoss, self).__init__()
        self.mask_ratio = mask_ratio
    
    def forward(self, pred_intensity, gt_intensity, mask=None):
        """
        Input:
            pred_intensity: (BS, 1, N, 1) 预测的强度
            gt_intensity: (BS, 1, N, 1) 真实强度
            mask: (BS, 1, N, 1) 可选的掩码，1表示要预测的点
        Output:
            loss: 标量
            prediction_error: (BS, 1, N, 1) 预测误差图
        """
        if mask is None:
            # 随机生成mask
            BS, _, N, _ = pred_intensity.shape
            mask = (torch.rand(BS, 1, N, 1, device=pred_intensity.device) < self.mask_ratio).float()
        
        # 只计算被mask点的损失
        error = (pred_intensity - gt_intensity).abs()
        masked_error = error * mask
        
        # 归一化
        loss = masked_error.sum() / (mask.sum() + 1e-6)
        
        return loss, error


class SpatialConsistencyLoss(nn.Module):
    """
    空间一致性损失
    
    思路：
    - 正常点云局部几何应该是平滑连续的
    - 通过KNN找到邻域，计算与邻域的偏离程度
    - 偏离大的点可能是噪声
    
    优化：为了避免内存溢出，使用采样策略
    """
    def __init__(self, k=8, sharpness=2.0, max_points=8000):
        super(SpatialConsistencyLoss, self).__init__()
        self.k = k
        self.sharpness = sharpness
        self.max_points = max_points  # 最大计算点数
    
    def forward(self, xyz, spatial_score):
        """
        Input:
            xyz: (BS, 3, N, 1) 点云坐标
            spatial_score: (BS, 1, N, 1) 预测的空间一致性分数
        Output:
            loss: 标量
        """
        BS, _, N, _ = xyz.shape
        xyz = xyz.squeeze(-1).permute(0, 2, 1)  # (BS, N, 3)
        spatial_score = spatial_score.squeeze(-1)  # (BS, 1, N)
        
        total_loss = 0.0
        
        for b in range(BS):
            xyz_b = xyz[b]  # (N, 3)
            score_b = spatial_score[b].squeeze(0)  # (N,)
            
            # 如果点数太多，随机采样
            if N > self.max_points:
                indices = torch.randperm(N, device=xyz.device)[:self.max_points]
                xyz_sampled = xyz_b[indices]  # (M, 3)
                score_sampled = score_b[indices]  # (M,)
            else:
                xyz_sampled = xyz_b
                score_sampled = score_b
            
            M = xyz_sampled.shape[0]
            
            # 计算采样点之间的距离
            dist = torch.cdist(xyz_sampled.unsqueeze(0), xyz_sampled.unsqueeze(0)).squeeze(0)  # (M, M)
            
            # 找K近邻
            _, knn_idx = dist.topk(self.k + 1, largest=False, dim=1)  # (M, k+1), 包含自己
            knn_idx = knn_idx[:, 1:]  # 去掉自己 (M, k)
            
            # 计算到邻域中心的距离
            knn_xyz = xyz_sampled[knn_idx]  # (M, k, 3)
            center_xyz = knn_xyz.mean(dim=1)  # (M, 3)
            dist_to_center = (xyz_sampled - center_xyz).norm(dim=1)  # (M,)
            
            # 归一化
            dist_to_center = dist_to_center / (dist_to_center.max() + 1e-6)
            
            # 一致性分数高的点，应该距离邻域中心近
            # 即：spatial_score 高 → dist_to_center 低
            consistency_loss = (score_sampled * dist_to_center).mean()
            
            total_loss = total_loss + consistency_loss
        
        return total_loss / BS


class DisplacementSparsityLoss(nn.Module):
    """
    位移稀疏损失
    
    思路：
    - 大部分点是正常的，位移应该接近0
    - 只有少数异常点需要大位移
    - 通过L1稀疏约束实现
    """
    def __init__(self, threshold=0.1):
        super(DisplacementSparsityLoss, self).__init__()
        self.threshold = threshold
    
    def forward(self, displacement, anomaly_prob):
        """
        Input:
            displacement: (BS, 3, N, 1) 预测的位移
            anomaly_prob: (BS, 1, N, 1) 异常概率
        Output:
            loss: 标量
        """
        # 位移的L2范数
        disp_norm = displacement.norm(dim=1, keepdim=True)  # (BS, 1, N, 1)
        
        # 正常点（低异常概率）的位移应该接近0
        normal_mask = (anomaly_prob < self.threshold).float()
        normal_disp_loss = (disp_norm * normal_mask).sum() / (normal_mask.sum() + 1e-6)
        
        # 异常点可以有大位移，但也要有上界
        anomaly_mask = (anomaly_prob >= self.threshold).float()
        anomaly_disp_loss = F.relu(disp_norm - 1.0) * anomaly_mask  # 位移超过1m才惩罚
        anomaly_disp_loss = anomaly_disp_loss.sum() / (anomaly_mask.sum() + 1e-6)
        
        return normal_disp_loss + 0.1 * anomaly_disp_loss


class ContrastiveLoss(nn.Module):
    """
    对比学习损失
    
    思路：
    - 正常点之间特征相似
    - 正常点与潜在异常点特征不同
    - 通过特征空间的对比学习增强判别能力
    
    优化：为了避免内存溢出，使用采样策略
    """
    def __init__(self, temperature=0.1, anomaly_threshold=0.5, max_points=8000):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.anomaly_threshold = anomaly_threshold
        self.max_points = max_points
    
    def forward(self, features, anomaly_prob):
        """
        Input:
            features: (BS, C, N, 1) 点特征
            anomaly_prob: (BS, 1, N, 1) 异常概率
        Output:
            loss: 标量
        """
        BS, C, N, _ = features.shape
        features = features.squeeze(-1).permute(0, 2, 1)  # (BS, N, C)
        anomaly_prob = anomaly_prob.squeeze(-1)  # (BS, 1, N)
        
        total_loss = 0.0
        valid_batches = 0
        
        for b in range(BS):
            feat_b = features[b]  # (N, C)
            prob_b = anomaly_prob[b].squeeze(0)  # (N,)
            
            # 分组：正常点 vs 潜在异常点
            normal_mask = prob_b < self.anomaly_threshold
            anomaly_mask = prob_b >= self.anomaly_threshold
            
            if normal_mask.sum() < 10 or anomaly_mask.sum() < 5:
                continue
            
            # 采样以减少内存
            normal_indices = torch.where(normal_mask)[0]
            anomaly_indices = torch.where(anomaly_mask)[0]
            
            # 采样正常点
            if len(normal_indices) > self.max_points // 2:
                normal_indices = normal_indices[torch.randperm(len(normal_indices), device=features.device)[:self.max_points // 2]]
            
            # 采样异常点
            if len(anomaly_indices) > self.max_points // 2:
                anomaly_indices = anomaly_indices[torch.randperm(len(anomaly_indices), device=features.device)[:self.max_points // 2]]
            
            # 合并并创建采样后的特征和mask
            all_indices = torch.cat([normal_indices, anomaly_indices])
            feat_sampled = feat_b[all_indices]  # (M, C)
            is_normal = torch.cat([torch.ones(len(normal_indices), device=features.device),
                                   torch.zeros(len(anomaly_indices), device=features.device)]).bool()
            
            # 归一化特征
            feat_sampled = F.normalize(feat_sampled, dim=1)
            
            # 计算采样点之间的相似度
            M = feat_sampled.shape[0]
            if M < 10:
                continue
                
            sim_matrix = torch.mm(feat_sampled, feat_sampled.t()) / self.temperature  # (M, M)
            
            # 正常点之间应该相似
            normal_sim = sim_matrix[is_normal][:, is_normal]
            if normal_sim.numel() > 1:
                # 去掉对角线
                mask = ~torch.eye(normal_sim.shape[0], dtype=bool, device=features.device)
                if mask.sum() > 0:
                    normal_sim_off_diag = normal_sim[mask]
                    positive_loss = -torch.log(torch.exp(normal_sim_off_diag).mean() + 1e-6).mean()
                else:
                    positive_loss = torch.tensor(0.0, device=features.device)
            else:
                positive_loss = torch.tensor(0.0, device=features.device)
            
            # 正常点与异常点应该不同
            cross_sim = sim_matrix[is_normal][:, ~is_normal]
            if cross_sim.numel() > 0:
                negative_loss = -torch.log(1 - torch.sigmoid(cross_sim) + 1e-6).mean()
            else:
                negative_loss = torch.tensor(0.0, device=features.device)
            
            total_loss = total_loss + positive_loss + negative_loss
            valid_batches += 1
        
        return total_loss / max(valid_batches, 1)


class IntensityDistributionLoss(nn.Module):
    """
    强度分布损失
    
    思路：
    - 雨雾噪声点的强度通常有特定的分布模式
    - 通过对抗学习让模型学习正常点的强度分布
    """
    def __init__(self):
        super(IntensityDistributionLoss, self).__init__()
    
    def forward(self, pred_intensity, gt_intensity, spatial_score):
        """
        Input:
            pred_intensity: (BS, 1, N, 1) 预测强度
            gt_intensity: (BS, 1, N, 1) 真实强度
            spatial_score: (BS, 1, N, 1) 空间一致性分数
        Output:
            loss: 标量
        """
        # 用空间一致性分数作为正常点的权重
        # 一致性高的点，强度预测应该更准确
        weight = spatial_score.detach()  # 不传梯度
        
        error = (pred_intensity - gt_intensity).abs()
        weighted_error = error * weight
        
        loss = weighted_error.mean()
        
        return loss


class SelfSupervisedDenoiseLoss(nn.Module):
    """
    综合自监督损失
    """
    def __init__(self, config):
        super(SelfSupervisedDenoiseLoss, self).__init__()
        
        self.intensity_loss = MaskedIntensityLoss(mask_ratio=config.get('mask_ratio', 0.15))
        self.spatial_loss = SpatialConsistencyLoss(
            k=config.get('knn_k', 8),
            sharpness=config.get('sharpness', 2.0)
        )
        self.sparsity_loss = DisplacementSparsityLoss(threshold=config.get('sparsity_threshold', 0.1))
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.get('temperature', 0.1),
            anomaly_threshold=config.get('anomaly_threshold', 0.5)
        )
        self.dist_loss = IntensityDistributionLoss()
        
        # 损失权重
        self.w_intensity = config.get('w_intensity', 1.0)
        self.w_spatial = config.get('w_spatial', 0.5)
        self.w_sparsity = config.get('w_sparsity', 0.3)
        self.w_contrastive = config.get('w_contrastive', 0.2)
        self.w_dist = config.get('w_dist', 0.5)
    
    def forward(self, pred_dict, pcds_xyz, pcds_intensity):
        """
        计算总损失
        
        Input:
            pred_dict: 网络预测结果字典
            pcds_xyz: (BS, 3, N, 1) 原始点云坐标
            pcds_intensity: (BS, 1, N, 1) 原始强度
        Output:
            loss_dict: 各损失项
        """
        loss_dict = {}
        
        # 1. 强度预测损失
        loss_intensity, pred_error = self.intensity_loss(
            pred_dict['pred_intensity'],
            pcds_intensity
        )
        loss_dict['loss_intensity'] = loss_intensity * self.w_intensity
        loss_dict['pred_error'] = pred_error  # 用于后续分析
        
        # 2. 空间一致性损失
        loss_spatial = self.spatial_loss(pcds_xyz, pred_dict['spatial_score'])
        loss_dict['loss_spatial'] = loss_spatial * self.w_spatial
        
        # 3. 位移稀疏损失
        loss_sparsity = self.sparsity_loss(
            pred_dict['displacement'],
            pred_dict['anomaly_prob']
        )
        loss_dict['loss_sparsity'] = loss_sparsity * self.w_sparsity
        
        # 4. 对比学习损失
        loss_contrastive = self.contrastive_loss(
            pred_dict['fused_feat'],
            pred_dict['anomaly_prob']
        )
        loss_dict['loss_contrastive'] = loss_contrastive * self.w_contrastive
        
        # 5. 强度分布损失
        loss_dist = self.dist_loss(
            pred_dict['pred_intensity'],
            pcds_intensity,
            pred_dict['spatial_score']
        )
        loss_dict['loss_dist'] = loss_dist * self.w_dist
        
        # 总损失
        total_loss = sum([v for k, v in loss_dict.items() if k.startswith('loss_')])
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
