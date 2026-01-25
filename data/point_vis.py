import open3d as o3d
import numpy as np
import os
import random

# --- 1. 论文风格高亮配色 (SemanticSTF Paper Style) ---
# 0号: 深灰色 (保留背景轮廓)
# 20号: 纯白色 (高亮显示雪花/雨滴噪点，论文图3a的效果)
PAPER_COLOR_MAP = {
    0:  [0.2, 0.2, 0.2], # unlabeled -> 深灰
    1:  [0.0, 1.0, 1.0], # car -> 青色 (Cyan)
    2:  [0.0, 1.0, 0.5], # bicycle -> 春绿
    3:  [1.0, 0.0, 1.0], # motorcycle -> 品红
    4:  [0.5, 0.0, 0.5], # truck -> 深紫
    5:  [0.0, 0.0, 1.0], # other-vehicle -> 蓝色
    6:  [1.0, 0.0, 0.0], # person -> 红色
    7:  [1.0, 0.4, 0.7], # bicyclist -> 粉色
    8:  [0.5, 0.2, 0.5], # motorcyclist -> 紫色
    9:  [1.0, 0.0, 1.0], # road -> 洋红色 (高对比度)
    10: [1.0, 0.6, 0.6], # parking -> 浅红
    11: [0.5, 0.0, 0.5], # sidewalk -> 深紫
    12: [0.5, 0.0, 0.0], # other-ground -> 暗红
    13: [1.0, 0.8, 0.0], # building -> 金黄色
    14: [1.0, 0.5, 0.0], # fence -> 橙色
    15: [0.0, 1.0, 0.0], # vegetation -> 纯绿
    16: [0.6, 0.3, 0.0], # trunk -> 棕色
    17: [0.7, 1.0, 0.0], # terrain -> 黄绿
    18: [1.0, 1.0, 0.8], # pole -> 浅黄
    19: [1.0, 0.0, 0.0], # traffic-sign -> 红
    20: [1.0, 1.0, 1.0], # invalid -> **纯白** (重点关注这个！)
}

class WeatherViewer:
    def __init__(self, dataset_root, train_txt, samples_per_weather=10):
        self.dataset_root = dataset_root
        self.velodyne_dir = os.path.join(dataset_root, "train/velodyne")
        self.label_dir = os.path.join(dataset_root, "train/labels")
        
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
        """解析 train.txt 并随机采样"""
        weather_dict = {'snow': [], 'rain': [], 'light_fog': [], 'dense_fog': []}
        
        if not os.path.exists(txt_path):
            print(f"Error: {txt_path} 不存在")
            return []

        print("正在解析 train.txt ...")
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    # parts[0] 是文件名, parts[1] 是天气
                    weather_dict[parts[1]].append(parts[0])
        
        playlist = []
        # 按顺序添加每种天气的样本，方便对比观察
        for weather in ['snow', 'rain', 'dense_fog', 'light_fog']:
            files = weather_dict.get(weather, [])
            if not files: continue
            
            # 随机抽样
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

        # --- 核心修改：Nx5 读取 ---
        try:
            raw_data = np.fromfile(bin_path, dtype=np.float32)
            # 这里的 5 就是你确认的维度 (x, y, z, intensity, ring)
            # 我们只取前 3 列 (x, y, z) 给 Open3D 显示
            points = raw_data.reshape((-1, 5))[:, :3]
        except Exception as e:
            print(f"  读取错误: {e}")
            # 如果 reshape 失败，可能是文件损坏或不是Nx5，返回 None 跳过
            return None, None, weather

        # 读取标签
        if os.path.exists(label_path):
            raw_labels = np.fromfile(label_path, dtype=np.uint32)
            labels = raw_labels & 0xFFFF # 只取低16位语义
        else:
            print("  警告: 找不到标签文件")
            labels = np.zeros(len(points), dtype=int)

        # 数量对齐 (防止bin和label点数不一致导致报错)
        if points.shape[0] != labels.shape[0]:
            print(f"  注意: 点数({points.shape[0]}) 与 标签数({labels.shape[0]}) 不匹配，已截断。")
            min_len = min(points.shape[0], labels.shape[0])
            points = points[:min_len]
            labels = labels[:min_len]

        # 上色
        colors = np.zeros((len(points), 3))
        # 默认设为深灰 (对应Label 0)
        colors[:] = PAPER_COLOR_MAP[0] 
        
        # 遍历字典填色
        for lbl, rgb in PAPER_COLOR_MAP.items():
            mask = (labels == lbl)
            if np.any(mask):
                colors[mask] = rgb
            
        return points, colors, weather

    def update_vis(self, vis):
        """刷新画面"""
        points, colors, weather = self.load_data(self.idx)
        
        if points is not None:
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(self.pcd)
            # 稍微重置下视角范围，防止切到空数据时视角乱飞
            # vis.reset_view_point(True) 
            print(f"  >>> 当前显示: {weather} (按 'D' 下一张)")
        else:
            print("  >>> 数据加载失败，请按下一张跳过")
            
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
        # 预加载第一帧
        points, colors, _ = self.load_data(0)
        if points is None:
            # 如果第一张就挂了，给个空数据防止崩溃
            points = np.zeros((1, 3))
            colors = np.zeros((1, 3))

        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # 创建窗口
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="SemanticSTF Viewer (Nx5 Fixed)", width=1280, height=720)
        
        # 渲染设置：纯黑背景，点大一点看起来更有“雷达感”
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0, 0.0]) 
        opt.point_size = 2.0 

        vis.add_geometry(self.pcd)

        # 注册按键
        vis.register_key_callback(262, self.next_frame) # 右箭头 ->
        vis.register_key_callback(68, self.next_frame)  # D ->
        vis.register_key_callback(263, self.prev_frame) # 左箭头 <-
        vis.register_key_callback(65, self.prev_frame)  # A <-

        print("\n" + "="*50)
        print("【操作说明】")
        print("  按 '→' (右箭头) 或 'D': 下一张")
        print("  按 '←' (左箭头) 或 'A': 上一张")
        print("  鼠标左键: 旋转 | 滚轮: 缩放 | Shift+左键: 平移")
        print("  按 'Q': 退出")
        print("="*50 + "\n")

        vis.run()
        vis.destroy_window()

if __name__ == "__main__":
    # ------------------------------------------------
    # 请确认你的路径是否正确
    # ------------------------------------------------
    dataset_path = "./SemanticSTF"
    train_file = "./SemanticSTF/train/train.txt" 
    
    viewer = WeatherViewer(dataset_path, train_file, samples_per_weather=10)
    if viewer.n_files > 0:
        viewer.run()