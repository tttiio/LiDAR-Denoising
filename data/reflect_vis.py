import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class IntensityViewer:
    def __init__(self, dataset_root, train_txt, samples_per_weather=10):
        self.dataset_root = dataset_root
        self.velodyne_dir = os.path.join(dataset_root, "train/velodyne")
        
        # 1. 准备播放列表
        self.playlist = self._prepare_playlist(train_txt, samples_per_weather)
        self.n_files = len(self.playlist)
        self.idx = 0
        
        # Open3D 点云对象
        self.pcd = o3d.geometry.PointCloud()
        
        if self.n_files == 0:
            print("错误: 未找到有效文件，请检查路径。")
            return

        print(f"准备就绪: 共加载 {self.n_files} 个场景")

    def _prepare_playlist(self, txt_path, limit):
        """解析 train.txt 并按天气分类采样"""
        weather_dict = {'snow': [], 'rain': [], 'light_fog': [], 'dense_fog': []}
        
        if not os.path.exists(txt_path):
            print(f"Error: {txt_path} 不存在")
            return []

        print("正在解析 train.txt ...")
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    weather_dict[parts[1]].append(parts[0])
        
        playlist = []
        for weather in ['snow', 'rain', 'dense_fog', 'light_fog']:
            files = weather_dict.get(weather, [])
            if not files: continue
            
            count = min(len(files), limit)
            selected = random.sample(files, count)
            for fname in selected:
                playlist.append((fname, weather))
        return playlist

    def load_data(self, idx):
        file_id, weather = self.playlist[idx]
        bin_path = os.path.join(self.velodyne_dir, f"{file_id}.bin")
        
        print(f"[{idx+1}/{self.n_files}] 天气: {weather} | 文件: {file_id}")

        try:
            raw_data = np.fromfile(bin_path, dtype=np.float32)
            # 5个维度: x, y, z, intensity, ring
            data = raw_data.reshape((-1, 5))
            
            points = data[:, :3]   
            intensity = data[:, 3] 
            
        except Exception as e:
            print(f"  读取错误: {e}")
            return None, None, weather

        # --- 颜色风格修改部分 ---
        
        # 1. 归一化强度
        # 论文中的图看起来对比度很高，我们将范围稍微压缩一下，让暗部更暗，亮部更亮
        # 通常 intensity 是 0-255，这里除以 255 归一化
        norm_intensity = intensity / 255.0
        
        # 增强对比度 (可选): 稍微做一点伽马校正，让中间调（橙色）更丰富
        # norm_intensity = np.power(norm_intensity, 0.8) 
        
        norm_intensity = np.clip(norm_intensity, 0.0, 1.0)
         
        # 2. 更改颜色映射表
        # 将 'jet' 改为 'hot' (黑->红->黄->白) 或 'afmhot' (更偏金黄)
        # 这正是您上传图片中的那种“热成像”风格
        cmap = plt.get_cmap('hot') 
        
        # 3. 生成颜色
        colors = cmap(norm_intensity)[:, :3]
            
        return points, colors, weather

    def update_vis(self, vis):
        points, colors, weather = self.load_data(self.idx)
        
        if points is not None:
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(self.pcd)
            print(f"  >>> 当前显示: {weather} (按 'D' 下一张)")
        return False

    def next_frame(self, vis):
        if self.idx < self.n_files - 1:
            self.idx += 1
            self.update_vis(vis)
        else:
            print("已经是最后一张了")
        return False

    def prev_frame(self, vis):
        if self.idx > 0:
            self.idx -= 1
            self.update_vis(vis)
        else:
            print("已经是第一张了")
        return False

    def run(self):
        # 预加载
        points, colors, _ = self.load_data(0)
        if points is None:
            points = np.zeros((1, 3))
            colors = np.zeros((1, 3))

        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="SemanticSTF Intensity (Hot Style)", width=1280, height=720)
        
        # 渲染设置：黑色背景 + 大点
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0]) 
        opt.point_size = 2.0 

        vis.add_geometry(self.pcd)

        # 注册按键
        vis.register_key_callback(262, self.next_frame) 
        vis.register_key_callback(68, self.next_frame)  
        vis.register_key_callback(263, self.prev_frame) 
        vis.register_key_callback(65, self.prev_frame)  

        print("\n" + "="*50)
        print("  按 '→' 或 'D': 下一张")
        print("  按 '←' 或 'A': 上一张")
        print("  按 'Q': 退出")
        print("="*50 + "\n")

        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    # --- 请确认路径 ---
    dataset_path = "./SemanticSTF"
    train_file = "./SemanticSTF/train/train.txt"
    
    viewer = IntensityViewer(dataset_path, train_file, samples_per_weather=10)
    if viewer.n_files > 0:
        viewer.run()