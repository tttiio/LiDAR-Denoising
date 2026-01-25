import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- 1. 语义分割配色 (Paper Style) ---
PAPER_COLOR_MAP = {
    0:  [0.2, 0.2, 0.2], # unlabeled -> 深灰
    1:  [0.0, 1.0, 1.0], # car -> 青色
    2:  [0.0, 1.0, 0.5], # bicycle
    3:  [1.0, 0.0, 1.0], # motorcycle
    4:  [0.5, 0.0, 0.5], # truck
    5:  [0.0, 0.0, 1.0], # other-vehicle
    6:  [1.0, 0.0, 0.0], # person
    7:  [1.0, 0.4, 0.7], # bicyclist
    8:  [0.5, 0.2, 0.5], # motorcyclist
    9:  [1.0, 0.0, 1.0], # road -> 洋红色
    10: [1.0, 0.6, 0.6], # parking
    11: [0.5, 0.0, 0.5], # sidewalk
    12: [0.5, 0.0, 0.0], # other-ground
    13: [1.0, 0.8, 0.0], # building
    14: [1.0, 0.5, 0.0], # fence
    15: [0.0, 1.0, 0.0], # vegetation
    16: [0.6, 0.3, 0.0], # trunk
    17: [0.7, 1.0, 0.0], # terrain
    18: [1.0, 1.0, 0.8], # pole
    19: [1.0, 0.0, 0.0], # traffic-sign
    20: [1.0, 1.0, 1.0], # invalid/noise -> 纯白 (重点!)
}

class DualViewer:
    def __init__(self, dataset_root, train_txt, samples_per_weather=10):
        self.dataset_root = dataset_root
        self.velodyne_dir = os.path.join(dataset_root, "train/velodyne")
        self.label_dir = os.path.join(dataset_root, "train/labels")
        
        # --- 【关键修改】加大偏移量 ---
        # KITTI/SemanticSTF 的左右范围大约是 [-50, 50] 米
        # 所以偏移量至少要 > 100 米才能完全分开
        # 这里设置为 150.0 米，保证中间有足够的空隙
        self.SHIFT_OFFSET = np.array([0.0, 150.0, 0.0]) 

        # 准备播放列表
        self.playlist = self._prepare_playlist(train_txt, samples_per_weather)
        self.n_files = len(self.playlist)
        self.idx = 0
        
        # 创建两个点云对象
        self.pcd_semantic = o3d.geometry.PointCloud()
        self.pcd_intensity = o3d.geometry.PointCloud()
        
        if self.n_files == 0:
            print("错误: 未找到有效文件，请检查路径。")
            return

        print(f"准备就绪: 共加载 {self.n_files} 个场景")

    def _prepare_playlist(self, txt_path, limit):
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
        label_path = os.path.join(self.label_dir, f"{file_id}.label")
        
        print(f"[{idx+1}/{self.n_files}] 天气: {weather} | 文件: {file_id}")

        try:
            raw_data = np.fromfile(bin_path, dtype=np.float32)
            data = raw_data.reshape((-1, 5))
            points = data[:, :3]      # XYZ
            intensity = data[:, 3]    # Intensity
        except Exception as e:
            print(f"  读取错误: {e}")
            return None, None, None, weather

        # --- 1. 处理语义颜色 ---
        if os.path.exists(label_path):
            raw_labels = np.fromfile(label_path, dtype=np.uint32)
            labels = raw_labels & 0xFFFF
        else:
            labels = np.zeros(len(points), dtype=int)

        min_len = min(points.shape[0], labels.shape[0])
        points = points[:min_len]
        intensity = intensity[:min_len]
        labels = labels[:min_len]

        sem_colors = np.zeros((len(points), 3))
        sem_colors[:] = PAPER_COLOR_MAP[0] 
        for lbl, rgb in PAPER_COLOR_MAP.items():
            mask = (labels == lbl)
            if np.any(mask):
                sem_colors[mask] = rgb

        # --- 2. 处理强度颜色 (热力图) ---
        norm_intensity = intensity / 255.0
        norm_intensity = np.power(np.clip(norm_intensity, 0, 1), 0.7)
        
        cmap = plt.get_cmap('hot')
        int_colors = cmap(norm_intensity)[:, :3]

        return points, sem_colors, int_colors, weather

    def update_vis(self, vis):
        points, sem_colors, int_colors, weather = self.load_data(self.idx)
        
        if points is not None:
            # 更新左侧点云 (语义)
            self.pcd_semantic.points = o3d.utility.Vector3dVector(points)
            self.pcd_semantic.colors = o3d.utility.Vector3dVector(sem_colors)
            
            # 更新右侧点云 (强度) - 使用更大的偏移量
            shifted_points = points + self.SHIFT_OFFSET
            self.pcd_intensity.points = o3d.utility.Vector3dVector(shifted_points)
            self.pcd_intensity.colors = o3d.utility.Vector3dVector(int_colors)
            
            vis.update_geometry(self.pcd_semantic)
            vis.update_geometry(self.pcd_intensity)
            
            if self.idx == 0:
                vis.reset_view_point(True)
                
            print(f"  >>> 显示: {weather}")
        return False

    def next_frame(self, vis):
        if self.idx < self.n_files - 1:
            self.idx += 1
            self.update_vis(vis)
        return False

    def prev_frame(self, vis):
        if self.idx > 0:
            self.idx -= 1
            self.update_vis(vis)
        return False

    def run(self):
        points, sem_colors, int_colors, _ = self.load_data(0)
        if points is None:
            points = np.zeros((1, 3))
            sem_colors = np.zeros((1, 3))
            int_colors = np.zeros((1, 3))

        self.pcd_semantic.points = o3d.utility.Vector3dVector(points)
        self.pcd_semantic.colors = o3d.utility.Vector3dVector(sem_colors)
        
        self.pcd_intensity.points = o3d.utility.Vector3dVector(points + self.SHIFT_OFFSET)
        self.pcd_intensity.colors = o3d.utility.Vector3dVector(int_colors)

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Dual View (Shift=150m)", width=1600, height=800)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0])
        opt.point_size = 2.0 

        vis.add_geometry(self.pcd_semantic)
        vis.add_geometry(self.pcd_intensity)

        vis.register_key_callback(262, self.next_frame) 
        vis.register_key_callback(68, self.next_frame)  
        vis.register_key_callback(263, self.prev_frame) 
        vis.register_key_callback(65, self.prev_frame)  

        print("\n" + "="*60)
        print(f"  已将偏移量调整为 150 米，应该不会重叠了。")
        print(f"  操作: 按 'D'/'→' 下一张, 'A'/'←' 上一张")
        print("="*60 + "\n")

        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    dataset_path = "./SemanticSTF"
    train_file = "./SemanticSTF/train/train.txt" 
    
    viewer = DualViewer(dataset_path, train_file, samples_per_weather=10)
    if viewer.n_files > 0:
        viewer.run()