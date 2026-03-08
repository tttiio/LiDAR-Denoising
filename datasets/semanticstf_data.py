"""
SemanticSTF 数据集加载器

特点：
1. 包含真实雨雾天气数据
2. 天气类型：rain, snow, light_fog, dense_fog
3. 用于自监督点云去噪训练
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
import random
from pathlib import Path


class SemanticSTFLoader(Dataset):
    """
    SemanticSTF 数据集加载器
    
    数据格式：
    - 点云: .bin 文件, (N, 5) -> [x, y, z, intensity, ring]
    - 标签: .label 文件, (N,) -> 语义标签
    - 天气: 从 split.txt 读取
    """
    
    # 天气类型映射
    WEATHER_TYPES = {
        'rain': 0,
        'snow': 1,
        'light_fog': 2,
        'dense_fog': 3,
        'clear': 4  # 如果有晴天数据
    }
    
    # 噪声点标签（SemanticSTF 中 unlabeled=0, invalid=20）
    NOISE_LABELS = [0, 20]
    
    def __init__(self, config, split='train', is_train=True):
        """
        Args:
            config: 配置对象
            split: 'train', 'val', 或 'test'
            is_train: 是否为训练模式
        """
        self.config = config
        self.split = split
        self.is_train = is_train
        self.frame_point_num = getattr(config, 'frame_point_num', 60000)
        self.Voxel = config.Voxel
        
        # 数据路径
        self.data_root = config.SeqDir
        self.split_dir = os.path.join(self.data_root, split)
        self.velodyne_dir = os.path.join(self.split_dir, 'velodyne')
        self.labels_dir = os.path.join(self.split_dir, 'labels')
        
        # 加载标签配置
        yaml_path = os.path.join(self.data_root, 'semanticstf.yaml')
        with open(yaml_path, 'r') as f:
            self.label_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 加载文件列表和天气类型
        self.sample_list = self._load_split_info()
        
        print(f'SemanticSTF {split} samples: {len(self.sample_list)}')
        self._print_weather_distribution()
        
        # 数据增强
        if is_train:
            self.aug_config = getattr(config, 'AugParam', None)
        else:
            self.aug_config = None
    
    def _load_split_info(self):
        """加载 split 信息，包含天气类型"""
        sample_list = []
        
        # 尝试读取 split.txt 文件
        split_file = os.path.join(self.split_dir, f'{self.split}.txt')
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ',' in line:
                        name, weather = line.split(',')
                    else:
                        name = line
                        weather = 'unknown'
                    
                    # 检查文件是否存在
                    pcd_path = os.path.join(self.velodyne_dir, f'{name}.bin')
                    label_path = os.path.join(self.labels_dir, f'{name}.label') if os.path.exists(self.labels_dir) else None
                    
                    if os.path.exists(pcd_path):
                        sample_list.append({
                            'name': name,
                            'pcd_path': pcd_path,
                            'label_path': label_path,
                            'weather': weather.strip() if isinstance(weather, str) else weather
                        })
        else:
            # 直接扫描 velodyne 目录
            for pcd_file in sorted(Path(self.velodyne_dir).glob('*.bin')):
                name = pcd_file.stem
                label_path = os.path.join(self.labels_dir, f'{name}.label')
                
                sample_list.append({
                    'name': name,
                    'pcd_path': str(pcd_file),
                    'label_path': label_path if os.path.exists(label_path) else None,
                    'weather': 'unknown'
                })
        
        return sample_list
    
    def _print_weather_distribution(self):
        """打印天气分布"""
        weather_count = {}
        for sample in self.sample_list:
            w = sample['weather']
            weather_count[w] = weather_count.get(w, 0) + 1
        
        print(f'Weather distribution:')
        for w, count in sorted(weather_count.items()):
            print(f'  {w}: {count}')
    
    def _load_point_cloud(self, pcd_path):
        """加载点云"""
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)
        return pcd  # (N, 5): [x, y, z, intensity, ring]
    
    def _load_labels(self, label_path):
        """加载标签"""
        if label_path and os.path.exists(label_path):
            labels = np.fromfile(label_path, dtype=np.uint32)
            return labels
        return None
    
    def _augment(self, pcd):
        """数据增强"""
        if self.aug_config is None:
            return pcd
        
        # 随机旋转
        if hasattr(self.aug_config, 'theta_range'):
            theta = np.random.uniform(
                self.aug_config.theta_range[0],
                self.aug_config.theta_range[1]
            ) * np.pi / 180.0
            
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rot_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            
            pcd[:, :3] = pcd[:, :3] @ rot_matrix.T
        
        # 随机平移
        if hasattr(self.aug_config, 'shift_range'):
            shift_x = np.random.uniform(*self.aug_config.shift_range[0])
            shift_y = np.random.uniform(*self.aug_config.shift_range[1])
            shift_z = np.random.uniform(*self.aug_config.shift_range[2])
            
            pcd[:, 0] += shift_x
            pcd[:, 1] += shift_y
            pcd[:, 2] += shift_z
        
        # 随机缩放
        if hasattr(self.aug_config, 'size_range'):
            scale = np.random.uniform(*self.aug_config.size_range)
            pcd[:, :3] *= scale
        
        # 添加噪声
        if hasattr(self.aug_config, 'noise_std') and self.aug_config.noise_std > 0:
            noise = np.random.normal(
                self.aug_config.noise_mean,
                self.aug_config.noise_std,
                pcd[:, :3].shape
            )
            pcd[:, :3] += noise
        
        return pcd
    
    def _quantize_coords(self, pcd):
        """坐标量化"""
        from . import utils
        
        coord = utils.Quantize(
            pcd,
            range_x=self.Voxel.range_x,
            range_y=self.Voxel.range_y,
            range_z=self.Voxel.range_z,
            size=self.Voxel.bev_shape
        )
        
        # PolarQuantize 的 range_y 参数是【径向距离范围】，不是Y轴范围
        # 径向距离 = sqrt(x² + y²)，范围应该是正数
        # PolarQuantize 需要 3 维 size (H, W, D)，但我们只用前两维用于 Range View
        rv_size = (self.Voxel.rv_shape[0], self.Voxel.rv_shape[1], 30)  # (H, W, D)
        sphere_coord = utils.PolarQuantize(
            pcd,
            phi_range=(-180.0, 180.0),  # 方位角范围
            range_y=(0.0, 70.0),        # 径向距离范围 (0到约100m，设70m覆盖大部分点)
            range_z=self.Voxel.range_z,  # 高度范围
            size=rv_size                 # 3 维 size
        )
        
        return coord, sphere_coord
    
    def _sample_points(self, pcd, labels=None):
        """采样点到固定数量"""
        N = pcd.shape[0]
        
        if self.frame_point_num is None:
            indices = np.arange(N)
        elif N >= self.frame_point_num:
            indices = np.random.choice(N, self.frame_point_num, replace=False)
        else:
            # 点数不够时重复采样
            indices = np.random.choice(N, self.frame_point_num, replace=True)
        
        pcd_sampled = pcd[indices]
        labels_sampled = labels[indices] if labels is not None else None
        
        return pcd_sampled, labels_sampled, indices
    
    def __getitem__(self, index):
        sample = self.sample_list[index]
        
        # 加载点云
        pcd = self._load_point_cloud(sample['pcd_path'])
        labels = self._load_labels(sample['label_path'])
        
        # 采样
        pcd, labels, indices = self._sample_points(pcd, labels)
        
        # 数据增强（仅训练时）
        if self.is_train:
            pcd = self._augment(pcd)
        
        # 分离 xyz 和 intensity
        xyz = pcd[:, :3]  # (N, 3)
        intensity = pcd[:, 3]  # (N,)
        ring = pcd[:, 4] if pcd.shape[1] > 4 else np.zeros(len(pcd))  # (N,)
        
        # 坐标量化
        coord, sphere_coord = self._quantize_coords(pcd)
        
        # 噪声掩码（用于评估，不用于训练）
        if labels is not None:
            noise_mask = np.isin(labels, self.NOISE_LABELS)
        else:
            noise_mask = np.zeros(len(pcd), dtype=bool)
        
        # 天气类型
        weather_type = self.WEATHER_TYPES.get(sample['weather'], -1)
        
        # 转为 tensor
        xyz_tensor = torch.FloatTensor(xyz.astype(np.float32)).transpose(0, 1).unsqueeze(-1)  # (3, N, 1)
        intensity_tensor = torch.FloatTensor(intensity.astype(np.float32)).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        # Quantize 返回 (N, 3)，但 VoxelMaxPool 只需要 (N, 2) 的 BEV 坐标
        coord_tensor = torch.FloatTensor(coord[:, :2].astype(np.float32)).unsqueeze(-1)  # (N, 2, 1)
        # PolarQuantize 返回 (N, 3)，但只需要前两维用于 Range View
        sphere_coord_tensor = torch.FloatTensor(sphere_coord[:, :2].astype(np.float32)).unsqueeze(-1)  # (N, 2, 1)
        noise_mask_tensor = torch.BoolTensor(noise_mask).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        
        # 处理 labels 为 None 的情况，生成一个全 -1 的占位符 Tensor
        if labels is not None:
            labels_tensor = torch.LongTensor(labels.astype(np.int64))
        else:
            # 打印警告，让你知道哪个文件缺了标签
            print(f"\n[Warning] 样本 {sample['name']} 缺失 label 文件，将使用 -1 填充！")
            labels_tensor = torch.full((len(xyz),), -1, dtype=torch.long)

        return {
            'xyz': xyz_tensor,                    
            'intensity': intensity_tensor,        
            'coord': coord_tensor,                
            'sphere_coord': sphere_coord_tensor,  
            'noise_mask': noise_mask_tensor,      
            'labels': labels_tensor,              # <--- 这里改成了安全的 Tensor
            'weather': weather_type,              
            'name': sample['name']                
        }
    
    def __len__(self):
        return len(self.sample_list)


# ==================== 训练/验证/测试加载器 ====================

class DataloadTrain(SemanticSTFLoader):
    """训练数据加载器"""
    def __init__(self, config):
        super(DataloadTrain, self).__init__(config, split='train', is_train=True)


class DataloadVal(SemanticSTFLoader):
    """验证数据加载器"""
    def __init__(self, config):
        super(DataloadVal, self).__init__(config, split='val', is_train=False)


class DataloadTest(SemanticSTFLoader):
    """测试数据加载器"""
    def __init__(self, config):
        super(DataloadTest, self).__init__(config, split='test', is_train=False)


# ==================== 天气感知的数据加载器 ====================

class WeatherAwareLoader(SemanticSTFLoader):
    """
    天气感知数据加载器
    
    特点：
    1. 可以按天气类型过滤数据
    2. 支持天气平衡采样
    """
    
    def __init__(self, config, split='train', weather_types=None, is_train=True):
        """
        Args:
            weather_types: 要加载的天气类型列表，如 ['rain', 'snow']
                         None 表示加载所有类型
        """
        self.weather_filter = weather_types
        super(WeatherAwareLoader, self).__init__(config, split=split, is_train=is_train)
    
    def _load_split_info(self):
        """加载并过滤数据"""
        all_samples = super()._load_split_info()
        
        if self.weather_filter is None:
            return all_samples
        
        # 按天气类型过滤
        filtered = [s for s in all_samples if s['weather'] in self.weather_filter]
        
        print(f'Filtered by weather types {self.weather_filter}: {len(filtered)} samples')
        
        return filtered


# ==================== 用于可视化的加载器 ====================

class VisualizationLoader(SemanticSTFLoader):
    """
    可视化数据加载器
    
    每种天气类型各加载若干样本
    """
    
    def __init__(self, config, samples_per_weather=5):
        self.samples_per_weather = samples_per_weather
        super(VisualizationLoader, self).__init__(config, split='test', is_train=False)
    
    def _load_split_info(self):
        all_samples = super()._load_split_info()
        
        # 按天气分组
        weather_groups = {}
        for sample in all_samples:
            w = sample['weather']
            if w not in weather_groups:
                weather_groups[w] = []
            weather_groups[w].append(sample)
        
        # 每种天气取若干样本
        selected = []
        for w, samples in weather_groups.items():
            n = min(self.samples_per_weather, len(samples))
            selected.extend(random.sample(samples, n))
        
        return selected
