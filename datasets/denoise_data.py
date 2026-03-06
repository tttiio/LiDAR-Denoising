"""
自监督点云去噪数据加载器

特点：
1. 不需要标注数据
2. 自动生成训练信号
3. 支持雨雾模拟增强
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import random
import os
import pickle as pkl
from nuscenes import NuScenes
from . import utils


class NoiseSimulator:
    """
    雨雾噪声模拟器
    
    模拟雨雾天气下LiDAR点云的噪声特征：
    1. 雨滴反射：强度异常高，位置随机
    2. 雾气散射：强度异常低，形成散点
    3. 多径反射：位置错误，强度异常
    """
    def __init__(self, rain_prob=0.3, fog_prob=0.2, multi_path_prob=0.1):
        self.rain_prob = rain_prob
        self.fog_prob = fog_prob
        self.multi_path_prob = multi_path_prob
    
    def simulate_rain_noise(self, points, intensity, noise_ratio=0.02):
        """
        模拟雨滴噪声
        
        特点：
        - 随机添加一些点
        - 强度较高
        - 位置在传感器上方
        """
        N = points.shape[0]
        noise_num = int(N * noise_ratio)
        
        # 生成随机噪声点
        noise_points = np.zeros((noise_num, 3))
        noise_points[:, 0] = np.random.uniform(-30, 30, noise_num)  # x
        noise_points[:, 1] = np.random.uniform(-30, 30, noise_num)  # y
        noise_points[:, 2] = np.random.uniform(-2, 5, noise_num)    # z (偏高)
        
        # 噪声强度：偏高
        noise_intensity = np.random.uniform(0.7, 1.0, noise_num)
        
        return noise_points, noise_intensity
    
    def simulate_fog_noise(self, points, intensity, noise_ratio=0.03):
        """
        模拟雾气噪声
        
        特点：
        - 散射形成的散点
        - 强度较低
        - 分布较均匀
        """
        N = points.shape[0]
        noise_num = int(N * noise_ratio)
        
        # 生成随机噪声点
        noise_points = np.random.uniform(-40, 40, (noise_num, 3))
        
        # 噪声强度：偏低
        noise_intensity = np.random.uniform(0.0, 0.3, noise_num)
        
        return noise_points, noise_intensity
    
    def simulate_multi_path_noise(self, points, intensity, noise_ratio=0.01):
        """
        模拟多径反射噪声
        
        特点：
        - 位置偏移
        - 强度异常
        - 常出现在反射面附近
        """
        N = points.shape[0]
        noise_num = int(N * noise_ratio)
        
        # 从现有点中选择一些点进行偏移
        if noise_num > 0 and N > 0:
            indices = np.random.choice(N, noise_num, replace=True)
            noise_points = points[indices].copy()
            
            # 添加随机偏移
            noise_points += np.random.uniform(-1, 1, (noise_num, 3))
            
            # 强度异常
            noise_intensity = np.random.uniform(0.3, 0.8, noise_num)
        else:
            noise_points = np.zeros((0, 3))
            noise_intensity = np.zeros(0)
        
        return noise_points, noise_intensity
    
    def add_noise(self, points, intensity):
        """
        添加混合噪声
        """
        noisy_points = points.copy()
        noisy_intensity = intensity.copy()
        noise_mask = np.zeros(len(points), dtype=bool)  # 标记哪些是噪声点
        
        # 雨滴噪声
        if random.random() < self.rain_prob:
            rain_points, rain_intensity = self.simulate_rain_noise(points, intensity)
            noisy_points = np.vstack([noisy_points, rain_points])
            noisy_intensity = np.concatenate([noisy_intensity, rain_intensity])
            noise_mask = np.concatenate([noise_mask, np.ones(len(rain_points), dtype=bool)])
        
        # 雾气噪声
        if random.random() < self.fog_prob:
            fog_points, fog_intensity = self.simulate_fog_noise(points, intensity)
            noisy_points = np.vstack([noisy_points, fog_points])
            noisy_intensity = np.concatenate([noisy_intensity, fog_intensity])
            noise_mask = np.concatenate([noise_mask, np.ones(len(fog_points), dtype=bool)])
        
        # 多径噪声
        if random.random() < self.multi_path_prob:
            mp_points, mp_intensity = self.simulate_multi_path_noise(points, intensity)
            if len(mp_points) > 0:
                noisy_points = np.vstack([noisy_points, mp_points])
                noisy_intensity = np.concatenate([noisy_intensity, mp_intensity])
                noise_mask = np.concatenate([noise_mask, np.ones(len(mp_points), dtype=bool)])
        
        return noisy_points, noisy_intensity, noise_mask


class DenoiseDataLoader(Dataset):
    """
    自监督去噪数据加载器
    """
    def __init__(self, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        
        # 噪声模拟器
        self.noise_simulator = NoiseSimulator() if is_train else None
        
        # 加载nuscenes配置
        with open('datasets/nuscenes.yaml', 'r') as f:
            self.task_cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        # 数据增强
        if is_train:
            self.aug = utils.DataAugment(
                noise_mean=config.AugParam.noise_mean,
                noise_std=config.AugParam.noise_std,
                theta_range=config.AugParam.theta_range,
                shift_range=config.AugParam.shift_range,
                size_range=config.AugParam.size_range
            )
        else:
            self.aug = None
        
        # 加载数据列表
        self.flist = []
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=config.SeqDir, verbose=True)
        
        fname_pkl = config.fname_pkl
        with open(fname_pkl, 'rb') as f:
            data_infos = pkl.load(f)['infos']
            for info in data_infos:
                lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
                fname_labels = os.path.join(
                    self.nusc.dataroot,
                    self.nusc.get('lidarseg', lidar_sd_token)['filename']
                )
                fname_pcds = os.path.join(
                    config.SeqDir,
                    '{}/{}/{}'.format(*info['lidar_path'].split('/')[-3:])
                )
                self.flist.append((fname_pcds, fname_labels, info['lidar_path']))
        
        print(f'{"Training" if is_train else "Validation"} Samples: {len(self.flist)}')
    
    def form_batch(self, pcds_xyzi, add_noise=False):
        """
        构建训练batch
        
        返回：
        - xyz: 笛卡尔坐标
        - intensity: 反射强度
        - coord: BEV坐标
        - sphere_coord: 极坐标
        - noise_mask: 噪声点标记（仅训练时）
        """
        # 分离xyz和强度
        xyz = pcds_xyzi[:, :3]
        intensity = pcds_xyzi[:, 3]
        
        # 模拟噪声
        noise_mask = None
        if add_noise and self.noise_simulator is not None:
            xyz, intensity, noise_mask = self.noise_simulator.add_noise(xyz, intensity)
        
        # 重新组合
        pcds_noisy = np.concatenate([xyz, intensity[:, np.newaxis]], axis=1)
        
        # 数据增强
        if self.aug is not None:
            pcds_noisy = self.aug(pcds_noisy)
        
        # 分离
        xyz = pcds_noisy[:, :3]
        intensity = pcds_noisy[:, 3]
        
        # 坐标量化
        coord = utils.Quantize(
            pcds_noisy,
            range_x=self.Voxel.range_x,
            range_y=self.Voxel.range_y,
            range_z=self.Voxel.range_z,
            size=self.Voxel.bev_shape
        )
        
        # PolarQuantize 需要正确的参数
        rv_size = (self.Voxel.rv_shape[0], self.Voxel.rv_shape[1], 30)  # (H, W, D) 3维
        sphere_coord = utils.PolarQuantize(
            pcds_noisy,
            phi_range=(-180.0, 180.0),
            range_y=(0.0, 70.0),  # 径向距离范围
            range_z=self.Voxel.range_z,
            size=rv_size
        )
        
        # 转为tensor - 只取前两维
        xyz = torch.FloatTensor(xyz.astype(np.float32)).transpose(0, 1).unsqueeze(-1)  # (3, N, 1)
        intensity = torch.FloatTensor(intensity.astype(np.float32)).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        coord_tensor = torch.FloatTensor(coord[:, :2].astype(np.float32)).unsqueeze(-1)  # (N, 2, 1)
        sphere_coord_tensor = torch.FloatTensor(sphere_coord[:, :2].astype(np.float32)).unsqueeze(-1)  # (N, 2, 1)
        
        if noise_mask is not None:
            noise_mask = torch.BoolTensor(noise_mask).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        
        return xyz, intensity, coord_tensor, sphere_coord_tensor, noise_mask
    
    def __getitem__(self, index):
        fname_pcds, fname_labels, seq_id = self.flist[index]
        
        # 加载点云
        pcds = np.fromfile(fname_pcds, dtype=np.float32, count=-1).reshape((-1, 5))[:, :4]
        
        # 采样
        if self.frame_point_num is not None:
            choice = np.random.choice(pcds.shape[0], self.frame_point_num, replace=True)
            pcds = pcds[choice]
        
        # 构建batch
        xyz, intensity, coord, sphere_coord, noise_mask = self.form_batch(
            pcds, 
            add_noise=self.is_train
        )
        
        return {
            'xyz': xyz,
            'intensity': intensity,
            'coord': coord,
            'sphere_coord': sphere_coord,
            'noise_mask': noise_mask,
            'seq_id': seq_id
        }
    
    def __len__(self):
        return len(self.flist)


class DataloadTrain(DenoiseDataLoader):
    """训练数据加载器"""
    def __init__(self, config):
        super(DataloadTrain, self).__init__(config, is_train=True)


class DataloadVal(DenoiseDataLoader):
    """验证数据加载器"""
    def __init__(self, config):
        super(DataloadVal, self).__init__(config, is_train=False)
