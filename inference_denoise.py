"""
点云去噪推理脚本

使用训练好的模型对点云进行去噪
"""

import torch
import numpy as np
import argparse
import os
import open3d as o3d
import matplotlib.pyplot as plt

from models_denoise import DenoiseNet
from datasets import utils
import importlib


def load_model(config_path, checkpoint_path, device):
    """加载模型"""
    config = importlib.import_module(config_path.replace('.py', '').replace('/', '.'))
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    model = DenoiseNet(pModel)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    return model, pModel


def preprocess_point_cloud(pcds, Voxel):
    """预处理点云"""
    # 假设输入是 (N, 4) 或 (N, 5)
    if pcds.shape[1] == 5:
        pcds = pcds[:, :4]
    
    xyz = pcds[:, :3]
    intensity = pcds[:, 3]
    
    # 坐标量化
    pcds_4d = np.concatenate([xyz, intensity[:, np.newaxis]], axis=1)
    
    coord = utils.Quantize(
        pcds_4d,
        range_x=Voxel.range_x,
        range_y=Voxel.range_y,
        range_z=Voxel.range_z,
        size=Voxel.bev_shape
    )
    
    sphere_coord = utils.PolarQuantize(
        pcds_4d,
        range_x=Voxel.range_x,
        range_y=Voxel.range_y,
        range_z=Voxel.range_z,
        size=Voxel.bev_shape
    )
    
    # 转为tensor
    xyz_tensor = torch.FloatTensor(xyz.astype(np.float32)).transpose(0, 1).unsqueeze(0).unsqueeze(-1)
    intensity_tensor = torch.FloatTensor(intensity.astype(np.float32)).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    coord_tensor = torch.FloatTensor(coord.astype(np.float32)).unsqueeze(0).unsqueeze(-1)
    sphere_coord_tensor = torch.FloatTensor(sphere_coord.astype(np.float32)).unsqueeze(0).unsqueeze(-1)
    
    return xyz_tensor, intensity_tensor, coord_tensor, sphere_coord_tensor


def denoise_point_cloud(model, pcds, Voxel, device, anomaly_threshold=0.5):
    """
    对点云进行去噪
    
    Args:
        model: 训练好的模型
        pcds: (N, 4) 点云数据 [x, y, z, intensity]
        Voxel: 体素配置
        device: 设备
        anomaly_threshold: 异常阈值，超过此值被认为是噪声点
    
    Returns:
        denoised_pcds: 去噪后的点云
        anomaly_mask: 异常点掩码
        anomaly_prob: 异常概率
    """
    # 预处理
    xyz, intensity, coord, sphere_coord = preprocess_point_cloud(pcds, Voxel)
    
    xyz = xyz.to(device)
    intensity = intensity.to(device)
    coord = coord.to(device)
    sphere_coord = sphere_coord.to(device)
    
    # 推理
    with torch.no_grad():
        denoised_xyz, anomaly_prob = model.infer(xyz, intensity, coord, sphere_coord)
    
    # 转回numpy
    denoised_xyz = denoised_xyz.squeeze().cpu().numpy().T  # (N, 3)
    anomaly_prob = anomaly_prob.squeeze().cpu().numpy()  # (N,)
    
    # 生成异常掩码
    anomaly_mask = anomaly_prob > anomaly_threshold
    
    # 去噪后的点云
    denoised_pcds = pcds.copy()
    denoised_pcds[:, :3] = denoised_xyz
    
    return denoised_pcds, anomaly_mask, anomaly_prob


def visualize_results(original_pcds, denoised_pcds, anomaly_mask, anomaly_prob, save_path=None):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 1. 原始点云 (BEV视角)
    ax = axes[0, 0]
    scatter = ax.scatter(
        original_pcds[:, 0], 
        original_pcds[:, 1],
        c=original_pcds[:, 3],
        s=0.5,
        cmap='viridis',
        alpha=0.5
    )
    ax.set_title('Original Point Cloud (BEV)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(scatter, ax=ax, label='Intensity')
    
    # 2. 去噪后点云
    ax = axes[0, 1]
    scatter = ax.scatter(
        denoised_pcds[:, 0],
        denoised_pcds[:, 1],
        c=denoised_pcds[:, 3],
        s=0.5,
        cmap='viridis',
        alpha=0.5
    )
    ax.set_title('Denoised Point Cloud (BEV)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(scatter, ax=ax, label='Intensity')
    
    # 3. 异常概率热力图
    ax = axes[1, 0]
    scatter = ax.scatter(
        original_pcds[:, 0],
        original_pcds[:, 1],
        c=anomaly_prob,
        s=0.5,
        cmap='hot',
        alpha=0.5
    )
    ax.set_title('Anomaly Probability')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(scatter, ax=ax, label='Anomaly Prob')
    
    # 4. 检测到的异常点
    ax = axes[1, 1]
    # 正常点
    normal_mask = ~anomaly_mask
    ax.scatter(
        original_pcds[normal_mask, 0],
        original_pcds[normal_mask, 1],
        c='blue',
        s=0.5,
        alpha=0.3,
        label='Normal'
    )
    # 异常点
    ax.scatter(
        original_pcds[anomaly_mask, 0],
        original_pcds[anomaly_mask, 1],
        c='red',
        s=2.0,
        alpha=0.8,
        label='Anomaly'
    )
    ax.set_title(f'Detected Anomalies ({anomaly_mask.sum()} points)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Saved visualization to {save_path}')
    else:
        plt.show()
    
    plt.close()


def save_point_cloud(pcds, save_path):
    """保存点云"""
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcds[:, :3])
    
    # 如果有强度信息，用作颜色
    if pcds.shape[1] >= 4:
        intensity = pcds[:, 3]
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        colors = plt.cm.viridis(intensity)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(save_path, pcd)
    print(f'Saved point cloud to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='LiDAR Point Cloud Denoising')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--input', type=str, required=True, help='Input point cloud file (.bin)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly threshold')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print('Loading model...')
    model, pModel = load_model(args.config, args.checkpoint, device)
    print('Model loaded!')
    
    # 加载点云
    print(f'Loading point cloud from {args.input}')
    pcds = np.fromfile(args.input, dtype=np.float32).reshape((-1, 5))[:, :4]
    print(f'Point cloud shape: {pcds.shape}')
    
    # 去噪
    print('Denoising...')
    denoised_pcds, anomaly_mask, anomaly_prob = denoise_point_cloud(
        model, pcds, pModel.Voxel, device, args.threshold
    )
    
    print(f'Detected {anomaly_mask.sum()} anomaly points ({anomaly_mask.sum() / len(pcds) * 100:.2f}%)')
    
    # 输出目录
    if args.output is None:
        args.output = os.path.dirname(args.input)
    os.makedirs(args.output, exist_ok=True)
    
    # 保存结果
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    
    # 保存去噪后的点云
    save_point_cloud(
        denoised_pcds[~anomaly_mask],  # 只保存正常点
        os.path.join(args.output, f'{input_name}_denoised.pcd')
    )
    
    # 保存异常点
    if anomaly_mask.sum() > 0:
        save_point_cloud(
            pcds[anomaly_mask],
            os.path.join(args.output, f'{input_name}_anomalies.pcd')
        )
    
    # 可视化
    if args.visualize:
        visualize_results(
            pcds, denoised_pcds, anomaly_mask, anomaly_prob,
            save_path=os.path.join(args.output, f'{input_name}_visualization.png')
        )
    
    print('Done!')


if __name__ == '__main__':
    main()
