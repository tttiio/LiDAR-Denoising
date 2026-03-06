"""
点云去噪评估脚本

在 SemanticSTF 测试集上评估去噪效果
使用标签中的噪声点（unlabeled=0, invalid=20）作为真值
"""

import os
import torch
import numpy as np
import argparse
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix
import seaborn as sns
import importlib

from models_denoise import DenoiseNet
from datasets.semanticstf_data import DataloadTest
from utils.self_supervised_loss import SelfSupervisedDenoiseLoss


def load_model(config_path, checkpoint_path, device):
    """加载模型"""
    config = importlib.import_module(config_path.replace('.py', '').replace('/', '.'))
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    model = DenoiseNet(pModel)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
    return model, pModel, pDataset


def evaluate_anomaly_detection(model, test_loader, device, threshold=0.5):
    """
    评估异常检测性能
    
    Returns:
        metrics: 包含各种评估指标
        all_results: 所有样本的详细结果
    """
    model.eval()
    
    all_probs = []      # 预测的异常概率
    all_labels = []     # 真实标签 (1=噪声, 0=正常)
    all_weather = []    # 天气类型
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc='Evaluating'):
            xyz = batch['xyz'].to(device)
            intensity = batch['intensity'].to(device)
            coord = batch['coord'].to(device)
            sphere_coord = batch['sphere_coord'].to(device)
            noise_mask = batch['noise_mask'].squeeze().cpu().numpy()  # True=噪声
            weather = batch['weather']
            
            # 推理
            pred_dict = model(xyz, intensity, coord, sphere_coord)
            anomaly_prob = pred_dict['anomaly_prob'].squeeze().cpu().numpy()
            
            all_probs.extend(anomaly_prob.flatten())
            all_labels.extend(noise_mask.flatten().astype(int))
            all_weather.extend([weather] * len(anomaly_prob.flatten()))
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_weather = np.array(all_weather)
    
    # 计算指标
    pred_binary = (all_probs > threshold).astype(int)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(all_labels, pred_binary).ravel()
    
    # 各种指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)  # 也叫 TPR
    specificity = tn / (tn + fp + 1e-8)  # TNR
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    # ROC 曲线和 AUC
    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # PR 曲线和 AUC
    precisions, recalls, thresholds_pr = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recalls, precisions)
    
    # 最佳阈值（最大化 F1）
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_threshold_idx] if best_threshold_idx < len(thresholds_pr) else threshold
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_threshold': best_threshold,
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    }
    
    all_results = {
        'probs': all_probs,
        'labels': all_labels,
        'weather': all_weather,
        'fpr': fpr,
        'tpr': tpr,
        'precisions': precisions,
        'recalls': recalls
    }
    
    return metrics, all_results


def evaluate_by_weather(all_results, weather_names=['rain', 'snow', 'light_fog', 'dense_fog']):
    """按天气类型分别评估"""
    probs = all_results['probs']
    labels = all_results['labels']
    weather = all_results['weather']
    
    weather_metrics = {}
    
    for w_idx, w_name in enumerate(weather_names):
        mask = weather == w_idx
        if mask.sum() == 0:
            continue
        
        w_probs = probs[mask]
        w_labels = labels[mask]
        
        # 计算该天气类型的指标
        pred_binary = (w_probs > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(w_labels, pred_binary, labels=[0, 1]).ravel()
        
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
        else:
            precision = recall = f1 = 0
        
        weather_metrics[w_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_points': len(w_labels),
            'num_noise': w_labels.sum(),
            'noise_ratio': w_labels.mean()
        }
    
    return weather_metrics


def plot_results(metrics, all_results, save_dir):
    """绘制评估结果图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(all_results['fpr'], all_results['tpr'], 'b-', linewidth=2, 
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(all_results['recalls'], all_results['precisions'], 'r-', linewidth=2,
             label=f'PR curve (AUC = {metrics["pr_auc"]:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 混淆矩阵
    cm = np.array([
        [metrics['confusion_matrix']['tn'], metrics['confusion_matrix']['fp']],
        [metrics['confusion_matrix']['fn'], metrics['confusion_matrix']['tp']]
    ])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Noise'],
                yticklabels=['Normal', 'Noise'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 异常概率分布
    probs = all_results['probs']
    labels = all_results['labels']
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(probs[labels == 0], bins=50, alpha=0.7, label='Normal', color='blue')
    plt.hist(probs[labels == 1], bins=50, alpha=0.7, label='Noise', color='red')
    plt.xlabel('Anomaly Probability')
    plt.ylabel('Count')
    plt.title('Anomaly Probability Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(probs[labels == 0], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    plt.hist(probs[labels == 1], bins=50, alpha=0.7, label='Noise', color='red', density=True)
    plt.xlabel('Anomaly Probability')
    plt.ylabel('Density')
    plt.title('Normalized Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'probability_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Saved plots to {save_dir}')


def save_metrics_report(metrics, weather_metrics, save_path):
    """保存评估报告"""
    with open(save_path, 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('Point Cloud Denoising Evaluation Report\n')
        f.write('=' * 60 + '\n\n')
        
        f.write('Overall Metrics:\n')
        f.write('-' * 40 + '\n')
        f.write(f"Accuracy:    {metrics['accuracy']:.4f}\n")
        f.write(f"Precision:   {metrics['precision']:.4f}\n")
        f.write(f"Recall:      {metrics['recall']:.4f}\n")
        f.write(f"Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"F1 Score:    {metrics['f1_score']:.4f}\n")
        f.write(f"ROC AUC:     {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC:      {metrics['pr_auc']:.4f}\n")
        f.write(f"Best Threshold: {metrics['best_threshold']:.4f}\n")
        f.write('\n')
        
        f.write('Confusion Matrix:\n')
        f.write('-' * 40 + '\n')
        f.write(f"True Negative:  {metrics['confusion_matrix']['tn']}\n")
        f.write(f"False Positive: {metrics['confusion_matrix']['fp']}\n")
        f.write(f"False Negative: {metrics['confusion_matrix']['fn']}\n")
        f.write(f"True Positive:  {metrics['confusion_matrix']['tp']}\n")
        f.write('\n')
        
        if weather_metrics:
            f.write('Metrics by Weather Type:\n')
            f.write('-' * 40 + '\n')
            for w_name, w_metrics in weather_metrics.items():
                f.write(f'\n{w_name.upper()}:\n')
                f.write(f"  Points: {w_metrics['num_points']}, Noise: {w_metrics['num_noise']} ({w_metrics['noise_ratio']*100:.2f}%)\n")
                f.write(f"  Precision: {w_metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {w_metrics['recall']:.4f}\n")
                f.write(f"  F1 Score:  {w_metrics['f1_score']:.4f}\n")
        
        f.write('\n' + '=' * 60 + '\n')
    
    print(f'Saved report to {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate Point Cloud Denoising')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly threshold')
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print('Loading model...')
    model, pModel, pDataset = load_model(args.config, args.checkpoint, device)
    print('Model loaded!')
    
    # 加载测试数据
    print('Loading test data...')
    test_dataset = DataloadTest(pDataset.Val)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    # 评估
    print('Evaluating...')
    metrics, all_results = evaluate_anomaly_detection(model, test_loader, device, args.threshold)
    
    # 按天气评估
    weather_metrics = evaluate_by_weather(all_results)
    
    # 输出结果
    print('\n' + '=' * 50)
    print('Evaluation Results:')
    print('=' * 50)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"PR AUC:      {metrics['pr_auc']:.4f}")
    print('=' * 50)
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    
    plot_results(metrics, all_results, args.output)
    save_metrics_report(metrics, weather_metrics, os.path.join(args.output, 'evaluation_report.txt'))
    
    print(f'\nEvaluation completed! Results saved to {args.output}')


if __name__ == '__main__':
    main()
