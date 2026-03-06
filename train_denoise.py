"""
自监督点云去噪网络训练脚本
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import time
import tqdm
import logging
import importlib

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models_denoise import DenoiseNet
from utils.self_supervised_loss import SelfSupervisedDenoiseLoss

# 根据配置选择数据加载器
def get_dataloader(data_src):
    if data_src == 'semanticstf_data':
        from datasets.semanticstf_data import DataloadTrain, DataloadVal, DataloadTest
        return DataloadTrain, DataloadVal, DataloadTest
    elif data_src == 'denoise_data':
        from datasets.denoise_data import DataloadTrain, DataloadVal
        return DataloadTrain, DataloadVal, None
    else:
        from datasets.denoise_data import DataloadTrain, DataloadVal
        return DataloadTrain, DataloadVal, None
from utils.logger import config_logger
from utils import builder

import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False


def reduce_tensor(inp):
    """Reduce tensor across all processes"""
    world_size = torch.distributed.get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp.clone()
        torch.distributed.reduce(reduced_inp, dst=0)
    return reduced_inp


def train_epoch(epoch, end_epoch, model, train_loader, optimizer, scheduler, 
                criterion, logger, log_frequency, device, fp16=True):
    """训练一个epoch"""
    model.train()
    rank = torch.distributed.get_rank()
    
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    
    for i, batch in tqdm.tqdm(enumerate(train_loader), desc=f'Epoch {epoch}'):
        # 获取数据
        xyz = batch['xyz'].to(device)
        intensity = batch['intensity'].to(device)
        coord = batch['coord'].to(device)
        sphere_coord = batch['sphere_coord'].to(device)
        
        # 前向传播
        if fp16:
            with torch.cuda.amp.autocast():
                pred_dict = model(xyz, intensity, coord, sphere_coord)
                loss_dict = criterion(pred_dict, xyz, intensity)
                loss = loss_dict['total_loss']
        else:
            pred_dict = model(xyz, intensity, coord, sphere_coord)
            loss_dict = criterion(pred_dict, xyz, intensity)
            loss = loss_dict['total_loss']
        
        # 反向传播
        optimizer.zero_grad()
        if fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # 日志
        if (i % log_frequency == 0) and rank == 0:
            log_str = f'Epoch: [{epoch}/{end_epoch}]; Iter: [{i}/{len(train_loader)}]; ' \
                      f'lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.6f}'
            
            for k, v in loss_dict.items():
                if k.startswith('loss_'):
                    log_str += f'; {k}: {v.item():.4f}'
            
            logger.info(log_str)
    
    return loss_dict


def validate(model, val_loader, criterion, device, logger):
    """验证"""
    model.eval()
    rank = torch.distributed.get_rank()
    
    total_losses = {}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc='Validating'):
            xyz = batch['xyz'].to(device)
            intensity = batch['intensity'].to(device)
            coord = batch['coord'].to(device)
            sphere_coord = batch['sphere_coord'].to(device)
            
            pred_dict = model(xyz, intensity, coord, sphere_coord)
            loss_dict = criterion(pred_dict, xyz, intensity)
            
            for k, v in loss_dict.items():
                if k not in total_losses:
                    total_losses[k] = 0
                if isinstance(v, torch.Tensor):
                    total_losses[k] += v.item()
            
            num_batches += 1
    
    # 平均
    for k in total_losses:
        total_losses[k] /= num_batches
    
    if rank == 0:
        log_str = 'Validation Results:'
        for k, v in total_losses.items():
            log_str += f' {k}: {v:.4f}'
        logger.info(log_str)
    
    return total_losses


def main(args, config):
    # 解析配置
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    # 损失配置
    pLoss = pModel.LossParam if hasattr(pModel, 'LossParam') else type('LossParam', (), {})()
    loss_config = {
        'mask_ratio': getattr(pLoss, 'mask_ratio', 0.15),
        'knn_k': getattr(pLoss, 'knn_k', 8),
        'sharpness': getattr(pLoss, 'sharpness', 2.0),
        'sparsity_threshold': getattr(pLoss, 'sparsity_threshold', 0.1),
        'temperature': getattr(pLoss, 'temperature', 0.1),
        'anomaly_threshold': getattr(pLoss, 'anomaly_threshold', 0.5),
        'w_intensity': getattr(pLoss, 'w_intensity', 1.0),
        'w_spatial': getattr(pLoss, 'w_spatial', 0.5),
        'w_sparsity': getattr(pLoss, 'w_sparsity', 0.3),
        'w_contrastive': getattr(pLoss, 'w_contrastive', 0.2),
        'w_dist': getattr(pLoss, 'w_dist', 0.5),
    }
    
    # 创建保存目录
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")
    os.makedirs(model_prefix, exist_ok=True)
    
    # 日志
    config_logger(os.path.join(save_path, "log.txt"))
    logger = logging.getLogger()
    
    # 分布式设置
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    # 随机种子
    seed = rank * pDataset.Train.num_workers + 50051
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 数据加载器
    DataloadTrainCls, DataloadValCls, _ = get_dataloader(pDataset.Train.data_src)
    
    train_dataset = DataloadTrainCls(pDataset.Train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=pGen.batch_size_per_gpu,
        shuffle=False,
        num_workers=pDataset.Train.num_workers,
        sampler=train_sampler,
        pin_memory=True
    )
    
    val_dataset = DataloadValCls(pDataset.Val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=pGen.batch_size_per_gpu,
        shuffle=False,
        num_workers=pDataset.Val.num_workers,
        pin_memory=True
    )
    
    print(f"Rank: {rank}/{world_size}; Batch size: {pGen.batch_size_per_gpu}")
    
    # 模型
    model = DenoiseNet(pModel)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(device),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    
    # 损失函数
    criterion = SelfSupervisedDenoiseLoss(loss_config)
    
    # 优化器
    optimizer = builder.get_optimizer(pOpt, model)
    scheduler = builder.get_scheduler(optimizer, pOpt, len(train_loader))
    
    if rank == 0:
        logger.info(f"Model: {model}")
        logger.info(f"Optimizer: {optimizer}")
        logger.info(f"Loss config: {loss_config}")
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(pOpt.schedule.begin_epoch, pOpt.schedule.end_epoch):
        train_sampler.set_epoch(epoch)
        
        # 训练
        train_loss = train_epoch(
            epoch, pOpt.schedule.end_epoch,
            model, train_loader, optimizer, scheduler,
            criterion, logger, pGen.log_frequency, device, pGen.fp16
        )
        
        # 验证
        if epoch % 5 == 0:
            val_loss = validate(model, val_loader, criterion, device, logger)
            
            # 保存最佳模型
            if rank == 0 and val_loss['total_loss'] < best_val_loss:
                best_val_loss = val_loss['total_loss']
                torch.save(
                    model.module.state_dict(),
                    os.path.join(model_prefix, 'best_model.pth')
                )
                logger.info(f'Saved best model at epoch {epoch}')
        
        # 定期保存
        if rank == 0 and epoch % 10 == 0:
            torch.save(
                model.module.state_dict(),
                os.path.join(model_prefix, f'model_epoch_{epoch}.pth')
            )
    
    # 保存最终模型
    if rank == 0:
        torch.save(
            model.module.state_dict(),
            os.path.join(model_prefix, 'final_model.pth')
        )
        logger.info('Training completed!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-supervised LiDAR Denoising')
    parser.add_argument('--config', help='config file path', type=str)
    args = parser.parse_args()
    
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)
