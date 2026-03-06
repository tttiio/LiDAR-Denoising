"""
LiDAR Point Cloud Denoising - Setup Script

自监督点云去噪系统，用于雨雾天气下的 LiDAR 点云数据处理

Installation:
    # 基础安装（仅依赖）
    pip install -r requirements.txt
    
    # 安装 CUDA 扩展（需要 NVIDIA GPU）
    python setup.py build_ext --inplace
    
    # 或者完整安装
    pip install -e .
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

# 检查是否有 CUDA
has_cuda = os.environ.get('FORCE_CUDA', '0') == '1'
if not has_cuda:
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except:
        has_cuda = False

# 定义扩展模块
ext_modules = [
    # CPU 扩展（始终编译）
    CppExtension(
        name='point_deep.cpu_kernel',
        sources=['pytorch_lib/src/point_deep.cpp'],
        include_dirs=['pytorch_lib/src']
    ),
]

# 如果有 CUDA，添加 CUDA 扩展
if has_cuda:
    ext_modules.append(
        CUDAExtension(
            name='point_deep.cuda_kernel',
            sources=[
                'pytorch_lib/src/point_deep_cuda.cpp',
                'pytorch_lib/src/point_deep_cuda_kernel.cu'
            ],
            include_dirs=['pytorch_lib/src'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    )
    print("CUDA detected, building CUDA extension...")
else:
    print("No CUDA detected, building CPU-only extension. Set FORCE_CUDA=1 to force CUDA build.")

setup(
    name='lidar_denoising',
    version='1.0.0',
    description='Self-supervised LiDAR point cloud denoising for rain and fog weather',
    author='LiDAR-Denoising',
    author_email='',
    url='',
    license='MIT',
    
    # 包发现
    packages=find_packages(exclude=['data', 'experiments', 'checkpoints']),
    
    # 依赖
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'scipy>=1.3.1',
        'PyYAML>=5.4.1',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.24.0',
        'tqdm>=4.50.0',
    ],
    
    # 额外依赖
    extras_require={
        'visualize': ['open3d>=0.13.0', 'seaborn>=0.11.0'],
        'nuscenes': ['nuscenes-devkit>=1.0.0'],
        'dev': ['tensorboard>=2.5.0', 'pytest'],
    },
    
    # CUDA/CPU 扩展
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    
    # 包含数据文件
    package_data={
        '': ['*.yaml', '*.json'],
    },
    
    # 入口点
    entry_points={
        'console_scripts': [
            'lidar-train=train_denoise:main',
            'lidar-evaluate=evaluate_denoise:main',
            'lidar-inference=inference_denoise:main',
        ],
    },
    
    # 分类
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    
    # 关键词
    keywords='lidar, denoising, point cloud, self-supervised, weather, autonomous driving',
)
