"""
测试脚本 - 验证数据加载和模型前向传播

快速检查代码是否能正常运行
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_data_loader():
    """测试数据加载器"""
    print("=" * 50)
    print("Testing Data Loader...")
    print("=" * 50)
    
    from datasets.semanticstf_data import DataloadTrain
    
    class MockConfig:
        SeqDir = './data/SemanticSTF'
        frame_point_num = 1000  # 小一点便于测试
        
        class Voxel:
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-5.0, 3.0)
            bev_shape = (600, 600, 30)
    
    try:
        dataset = DataloadTrain(MockConfig)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"✓ Sample loaded:")
        print(f"  - xyz shape: {sample['xyz'].shape}")
        print(f"  - intensity shape: {sample['intensity'].shape}")
        print(f"  - coord shape: {sample['coord'].shape}")
        print(f"  - sphere_coord shape: {sample['sphere_coord'].shape}")
        print(f"  - noise_mask shape: {sample['noise_mask'].shape}")
        print(f"  - weather: {sample['weather']}")
        print(f"  - name: {sample['name']}")
        
        return True, sample
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model():
    """测试模型"""
    print("\n" + "=" * 50)
    print("Testing Model...")
    print("=" * 50)
    
    from models_denoise import DenoiseNet
    
    class MockModelConfig:
        class Voxel:
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-5.0, 3.0)
            bev_shape = (600, 600, 30)
            rv_shape = (64, 2048)
        
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
    
    try:
        model = DenoiseNet(MockModelConfig)
        print(f"✓ Model created")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        return True, model
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model, sample):
    """测试前向传播"""
    print("\n" + "=" * 50)
    print("Testing Forward Pass...")
    print("=" * 50)
    
    try:
        model.eval()
        
        # 准备输入
        xyz = sample['xyz'].unsqueeze(0)  # 添加 batch 维度
        intensity = sample['intensity'].unsqueeze(0)
        coord = sample['coord'].unsqueeze(0)
        sphere_coord = sample['sphere_coord'].unsqueeze(0)
        
        print(f"Input shapes:")
        print(f"  - xyz: {xyz.shape}")
        print(f"  - intensity: {intensity.shape}")
        print(f"  - coord: {coord.shape}")
        print(f"  - sphere_coord: {sphere_coord.shape}")
        
        # 前向传播
        with torch.no_grad():
            pred_dict = model(xyz, intensity, coord, sphere_coord)
        
        print(f"\n✓ Forward pass successful!")
        print(f"Output shapes:")
        for key, value in pred_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")
        
        # 检查输出范围
        print(f"\nOutput ranges:")
        print(f"  - anomaly_prob: [{pred_dict['anomaly_prob'].min():.4f}, {pred_dict['anomaly_prob'].max():.4f}]")
        print(f"  - spatial_score: [{pred_dict['spatial_score'].min():.4f}, {pred_dict['spatial_score'].max():.4f}]")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss():
    """测试损失函数"""
    print("\n" + "=" * 50)
    print("Testing Loss Function...")
    print("=" * 50)
    
    from utils.self_supervised_loss import SelfSupervisedDenoiseLoss
    
    try:
        loss_config = {
            'mask_ratio': 0.15,
            'knn_k': 8,
            'w_intensity': 1.0,
            'w_spatial': 0.5,
        }
        
        criterion = SelfSupervisedDenoiseLoss(loss_config)
        print(f"✓ Loss function created")
        
        return True, criterion
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    print("Starting tests...")
    print()
    
    # 测试数据加载
    data_ok, sample = test_data_loader()
    if not data_ok:
        print("\n⚠ Data loader test failed, skipping remaining tests")
        return
    
    # 测试模型创建
    model_ok, model = test_model()
    if not model_ok:
        print("\n⚠ Model test failed, skipping forward pass test")
        return
    
    # 测试前向传播
    forward_ok = test_forward_pass(model, sample)
    
    # 测试损失函数
    loss_ok, criterion = test_loss()
    
    # 总结
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    print(f"Data Loader:    {'✓ PASS' if data_ok else '✗ FAIL'}")
    print(f"Model Creation: {'✓ PASS' if model_ok else '✗ FAIL'}")
    print(f"Forward Pass:   {'✓ PASS' if forward_ok else '✗ FAIL'}")
    print(f"Loss Function:  {'✓ PASS' if loss_ok else '✗ FAIL'}")
    print("=" * 50)
    
    if all([data_ok, model_ok, forward_ok, loss_ok]):
        print("\n🎉 All tests passed! Ready to train.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")


if __name__ == '__main__':
    main()
