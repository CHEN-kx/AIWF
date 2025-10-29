# AIWx - Reference Forecast Evaluation Script
"""
参考预报方法评估脚本

用于评估和对比不同参考预报方法的性能，为AI模型提供性能基准。
"""

import os
import torch
import numpy as np
import torch.utils.data as data
from tqdm import tqdm

from src.dataset.dataset_pangu import Dataset_pangu
from src.tools.reference_forecasts import Persistence, Climatology, WeeklyClimatology
from src.tools.metrics import create_latitude_weights


class ReferenceForecastEvaluator:
    """
    参考预报方法评估器
    
    提供统一接口评估不同参考预报方法，生成详细的性能报告。
    """
    
    def __init__(self, data_dir='/datadir', device='cuda'):
        """
        初始化评估器
        
        Args:
            data_dir: 数据根目录
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.data_dir = data_dir
        self.device = device
        
        # 初始化参考预报方法
        self.persistence = Persistence()
        self.climatology = None
        self.weekly_climatology = None
        
        print(f"Reference Forecast Evaluator initialized")
        print(f"  Device: {device}")
        print(f"  Data dir: {data_dir}")
    
    def setup_climatology(self, years=range(1979, 2015), 
                         precomputed_path=None,
                         save_path='climatology.npy'):
        """
        设置气候态方法
        
        Args:
            years: 用于计算的年份范围
            precomputed_path: 预计算文件路径
            save_path: 保存路径
        """
        self.climatology = Climatology(
            data_dir=self.data_dir,
            years=years,
            precomputed_path=precomputed_path
        )
        
        if self.climatology.climatology is None:
            print("\nClimatology not found, computing...")
            print("This may take several hours on first run.")
            self.climatology.compute(save_path=save_path)
    
    def setup_weekly_climatology(self, years=range(1979, 2015),
                                precomputed_path=None,
                                save_path='weekly_climatology.npy'):
        """
        设置周气候态方法
        
        Args:
            years: 用于计算的年份范围
            precomputed_path: 预计算文件路径
            save_path: 保存路径
        """
        self.weekly_climatology = WeeklyClimatology(
            data_dir=self.data_dir,
            years=years,
            precomputed_path=precomputed_path
        )
        
        if self.weekly_climatology.weekly_climatology is None:
            print("\nWeekly climatology not found, computing...")
            print("This may take several hours and requires ~100GB memory.")
            self.weekly_climatology.compute(save_path=save_path)
    
    def compute_rmse_per_variable(self, predictions, targets):
        """
        计算每个变量的RMSE
        
        Args:
            predictions: (batch, time, 69, 721, 1440)
            targets: (batch, time, 69, 721, 1440)
        
        Returns:
            rmse_dict: 每个变量的RMSE字典
        """
        # 变量名列表
        var_names = []
        pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        
        for var in ['z', 'q', 't', 'u', 'v']:
            for level in pressure_levels:
                var_names.append(f'{var}{level}')
        var_names.extend(['msl', 'u10', 'v10', 't2m'])
        
        rmse_dict = {}
        weights = torch.from_numpy(create_latitude_weights(721)).to(predictions.device)
        
        for i, var_name in enumerate(var_names):
            pred_var = predictions[:, :, i, :, :]
            target_var = targets[:, :, i, :, :]
            
            # 纬度加权RMSE
            error = ((pred_var - target_var) ** 2).mean(3)  # 经度平均
            error = torch.sqrt((error * weights).mean(2))    # 纬度加权平均
            rmse = error.mean().item()
            
            rmse_dict[var_name] = rmse
        
        return rmse_dict
    
    def evaluate_method(self, method, dataloader, method_name='Reference Method'):
        """
        评估单个参考预报方法
        
        Args:
            method: 参考预报方法对象
            dataloader: 数据加载器
            method_name: 方法名称（用于显示）
        
        Returns:
            results: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {method_name}")
        print(f"{'='*60}")
        
        all_predictions = []
        all_targets = []
        lead_time_scores = {f"{(i+1)*6}h": [] for i in range(20)}
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=method_name)):
            # 移到设备
            input_air, input_surface = inputs
            input_air = input_air.to(self.device)
            input_surface = input_surface.to(self.device)
            targets = targets.to(self.device)
            
            # 重塑targets（如果需要）
            if len(targets.shape) == 3:
                batch_size = targets.shape[0]
                n_timesteps = 20
                targets = targets.view(batch_size, n_timesteps, 69, 721, 1440)
            
            # 生成预测
            with torch.no_grad():
                if method_name == "Weekly Climatology":
                    # 周气候态需要指定初始周次，这里默认使用0
                    predictions = method((input_air, input_surface), n_steps=20, initial_week=0)
                else:
                    predictions = method((input_air, input_surface), n_steps=20)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            # 按预报时效计算RMSE
            weights = torch.from_numpy(create_latitude_weights(721)).to(predictions.device)
            for t in range(20):
                pred_t = predictions[:, t, :, :, :]
                target_t = targets[:, t, :, :, :]
                
                # 纬度加权RMSE
                error = ((pred_t - target_t) ** 2).mean(3)
                error = torch.sqrt((error * weights).mean(2))
                rmse = error.mean().item()
                
                lead_time_scores[f"{(t+1)*6}h"].append(rmse)
        
        # 汇总结果
        results = {
            'method_name': method_name,
            'mean_rmse': np.mean([np.mean(scores) for scores in lead_time_scores.values()]),
            'lead_time_rmse': {k: np.mean(v) for k, v in lead_time_scores.items()},
        }
        
        # 计算部分批次的逐变量RMSE
        if len(all_predictions) > 0:
            sample_pred = torch.cat(all_predictions[:5], dim=0)
            sample_tgt = torch.cat(all_targets[:5], dim=0)
            var_rmse = self.compute_rmse_per_variable(sample_pred, sample_tgt)
            results['variable_rmse'] = var_rmse
        
        # 打印结果
        print(f"\nOverall Mean RMSE: {results['mean_rmse']:.4f}")
        print(f"\nRMSE by Lead Time:")
        for lead_time, rmse in results['lead_time_rmse'].items():
            print(f"  {lead_time:>5s}: {rmse:.4f}")
        
        return results
    
    def compare_all_methods(self, dataloader):
        """
        评估并对比所有可用的参考预报方法
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            comparison: 所有方法的评估结果
        """
        comparison = {}
        
        # 评估Persistence（总是可用）
        results = self.evaluate_method(self.persistence, dataloader, "Persistence")
        comparison['persistence'] = results
        
        # 评估Climatology（如果可用）
        if self.climatology is not None:
            results = self.evaluate_method(self.climatology, dataloader, "Climatology")
            comparison['climatology'] = results
        
        # 评估Weekly Climatology（如果可用）
        if self.weekly_climatology is not None:
            results = self.evaluate_method(
                self.weekly_climatology, dataloader, "Weekly Climatology"
            )
            comparison['weekly_climatology'] = results
        
        # 打印对比摘要
        print(f"\n{'='*60}")
        print("REFERENCE FORECAST COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Method':<30s} {'Mean RMSE':>12s}")
        print(f"{'-'*60}")
        for method, result in comparison.items():
            print(f"{result['method_name']:<30s} {result['mean_rmse']:>12.4f}")
        
        return comparison


def main():
    """主评估函数"""
    # 配置
    DATA_DIR = '/datadir'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 2
    NUM_WORKERS = 2
    
    print(f"Using device: {DEVICE}")
    
    # 初始化评估器
    evaluator = ReferenceForecastEvaluator(data_dir=DATA_DIR, device=DEVICE)
    
    # 加载验证数据
    print("\nLoading validation dataset...")
    val_dataset = Dataset_pangu(split='val')
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # 设置气候态方法（可选，首次会自动计算）
    # evaluator.setup_climatology(
    #     years=range(1979, 2015),
    #     precomputed_path='climatology.npy'
    # )
    
    # evaluator.setup_weekly_climatology(
    #     years=range(1979, 2015),
    #     precomputed_path='weekly_climatology.npy'
    # )
    
    # 运行评估
    results = evaluator.compare_all_methods(val_loader)
    
    # 保存结果
    save_path = 'reference_forecast_evaluation_results.npy'
    np.save(save_path, results)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()