# AIWx - Flexible Reference Forecast Evaluation
"""
灵活的参考预报评估脚本

支持配置化的数据形状，可以轻松适配不同的数据集。
"""

import os
import torch
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
from typing import Optional, Dict, List

from src.tools.reference_forecasts import Persistence, Climatology, WeeklyClimatology
from src.tools.improved_metrics import WeatherMetrics, DataConfig


class FlexibleReferenceForecastEvaluator:
    """
    灵活的参考预报评估器
    
    支持配置化的数据形状和多种评估指标。
    """
    
    def __init__(self,
                 data_config: DataConfig = None,
                 data_dir: str = '/datadir',
                 device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            data_config: 数据配置，如果为None则使用Pangu默认配置
            data_dir: 数据根目录
            device: 运行设备
        
        Examples:
            # 使用Pangu配置
            >>> evaluator = FlexibleReferenceForecastEvaluator()
            
            # 使用ERA5配置
            >>> config = DataConfig.from_era5_1deg(n_channels=50)
            >>> evaluator = FlexibleReferenceForecastEvaluator(data_config=config)
            
            # 使用自定义配置
            >>> config = DataConfig(n_channels=100, n_lat=361, n_lon=720)
            >>> evaluator = FlexibleReferenceForecastEvaluator(data_config=config)
        """
        self.data_config = data_config or DataConfig.from_pangu()
        self.data_dir = data_dir
        self.device = device
        
        # 初始化metrics计算器
        self.metrics = WeatherMetrics(self.data_config)
        
        # 初始化参考预报方法
        self.persistence = Persistence()
        self.climatology = None
        self.weekly_climatology = None
        
        print(f"Flexible Reference Forecast Evaluator initialized")
        print(f"  Device: {device}")
        print(f"  Data config: C={self.data_config.n_channels}, "
              f"H={self.data_config.n_lat}, W={self.data_config.n_lon}")
        print(f"  Data dir: {data_dir}")
    
    def setup_climatology(self,
                         years=range(1979, 2015),
                         precomputed_path: Optional[str] = None,
                         save_path: str = 'climatology.npy'):
        """设置气候态方法"""
        self.climatology = Climatology(
            data_dir=self.data_dir,
            years=years,
            precomputed_path=precomputed_path
        )
        
        if self.climatology.climatology is None:
            print("\nClimatology not found, computing...")
            self.climatology.compute(save_path=save_path)
    
    def setup_weekly_climatology(self,
                                years=range(1979, 2015),
                                precomputed_path: Optional[str] = None,
                                save_path: str = 'weekly_climatology.npy'):
        """设置周气候态方法"""
        self.weekly_climatology = WeeklyClimatology(
            data_dir=self.data_dir,
            years=years,
            precomputed_path=precomputed_path
        )
        
        if self.weekly_climatology.weekly_climatology is None:
            print("\nWeekly climatology not found, computing...")
            self.weekly_climatology.compute(save_path=save_path)
    
    def _reshape_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        重塑targets张量以匹配预期格式
        
        Args:
            targets: 可能的形状:
                - (batch, time*C, H, W) 需要重塑
                - (batch, time, C, H, W) 已经正确
        
        Returns:
            targets: (batch, time, C, H, W)
        """
        if len(targets.shape) == 4:  # (batch, time*C, H, W)
            batch_size = targets.shape[0]
            n_timesteps = 20  # 假设20个时间步
            targets = targets.view(
                batch_size,
                n_timesteps,
                self.data_config.n_channels,
                self.data_config.n_lat,
                self.data_config.n_lon
            )
        
        return targets
    
    def evaluate_method(self,
                       method,
                       dataloader,
                       method_name: str = 'Reference Method',
                       compute_per_variable: bool = False) -> Dict:
        """
        评估单个参考预报方法
        
        Args:
            method: 参考预报方法对象
            dataloader: 数据加载器
            method_name: 方法名称
            compute_per_variable: 是否计算每个变量的指标
        
        Returns:
            results: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {method_name}")
        print(f"{'='*60}")
        
        all_predictions = []
        all_targets = []
        
        # 按时效统计
        lead_time_rmse = {f"{(i+1)*6}h": [] for i in range(20)}
        lead_time_mae = {f"{(i+1)*6}h": [] for i in range(20)}
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc=method_name)):
            # 移到设备
            input_air, input_surface = inputs
            input_air = input_air.to(self.device)
            input_surface = input_surface.to(self.device)
            targets = targets.to(self.device)
            
            # 重塑targets
            targets = self._reshape_targets(targets)
            
            # 生成预测
            with torch.no_grad():
                if method_name == "Weekly Climatology":
                    predictions = method((input_air, input_surface), n_steps=20, initial_week=0)
                else:
                    predictions = method((input_air, input_surface), n_steps=20)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            # 按时效计算指标
            for t in range(20):
                pred_t = predictions[:, t, :, :, :]  # (batch, C, H, W)
                target_t = targets[:, t, :, :, :]
                
                # RMSE
                rmse = self.metrics.compute_rmse(
                    pred_t, target_t,
                    latitude_weighted=True,
                    reduction='mean'
                )
                lead_time_rmse[f"{(t+1)*6}h"].append(rmse.item())
                
                # MAE
                mae = self.metrics.compute_mae(
                    pred_t, target_t,
                    latitude_weighted=True,
                    reduction='mean'
                )
                lead_time_mae[f"{(t+1)*6}h"].append(mae.item())
        
        # 汇总结果
        results = {
            'method_name': method_name,
            'mean_rmse': np.mean([np.mean(scores) for scores in lead_time_rmse.values()]),
            'mean_mae': np.mean([np.mean(scores) for scores in lead_time_mae.values()]),
            'lead_time_rmse': {k: np.mean(v) for k, v in lead_time_rmse.items()},
            'lead_time_mae': {k: np.mean(v) for k, v in lead_time_mae.items()},
        }
        
        # 按变量计算（可选）
        if compute_per_variable and len(all_predictions) > 0:
            sample_pred = torch.cat(all_predictions[:5], dim=0)
            sample_tgt = torch.cat(all_targets[:5], dim=0)
            var_rmse = self._compute_per_variable_rmse(sample_pred, sample_tgt)
            results['variable_rmse'] = var_rmse
        
        # 打印结果
        print(f"\nOverall Mean RMSE: {results['mean_rmse']:.4f}")
        print(f"Overall Mean MAE:  {results['mean_mae']:.4f}")
        
        print(f"\nRMSE by Lead Time:")
        for lead_time, rmse in list(results['lead_time_rmse'].items())[:5]:
            mae = results['lead_time_mae'][lead_time]
            print(f"  {lead_time:>5s}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        print(f"  ...")
        
        return results
    
    def _compute_per_variable_rmse(self,
                                   predictions: torch.Tensor,
                                   targets: torch.Tensor) -> Dict[str, float]:
        """
        计算每个变量的RMSE
        
        Args:
            predictions: (batch, time, C, H, W)
            targets: (batch, time, C, H, W)
        
        Returns:
            var_rmse: 每个变量的RMSE字典
        """
        # 这里需要知道变量名，对于不同数据集可能不同
        # 暂时使用通用命名
        var_rmse = {}
        
        for c in range(self.data_config.n_channels):
            pred_c = predictions[:, :, c, :, :]
            target_c = targets[:, :, c, :, :]
            
            # 计算该变量的RMSE
            rmse = self.metrics.compute_rmse(
                pred_c.unsqueeze(2),  # 添加C维度
                target_c.unsqueeze(2),
                latitude_weighted=True,
                reduction='mean'
            )
            
            var_rmse[f'var_{c:02d}'] = rmse.item()
        
        return var_rmse
    
    def compare_all_methods(self,
                           dataloader,
                           compute_per_variable: bool = False) -> Dict:
        """
        评估并对比所有可用的参考预报方法
        
        Args:
            dataloader: 数据加载器
            compute_per_variable: 是否计算每个变量的指标
        
        Returns:
            comparison: 所有方法的评估结果
        """
        comparison = {}
        
        # 评估Persistence
        results = self.evaluate_method(
            self.persistence,
            dataloader,
            "Persistence",
            compute_per_variable
        )
        comparison['persistence'] = results
        
        # 评估Climatology（如果可用）
        if self.climatology is not None:
            results = self.evaluate_method(
                self.climatology,
                dataloader,
                "Climatology",
                compute_per_variable
            )
            comparison['climatology'] = results
        
        # 评估Weekly Climatology（如果可用）
        if self.weekly_climatology is not None:
            results = self.evaluate_method(
                self.weekly_climatology,
                dataloader,
                "Weekly Climatology",
                compute_per_variable
            )
            comparison['weekly_climatology'] = results
        
        # 打印对比摘要
        print(f"\n{'='*60}")
        print("REFERENCE FORECAST COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Method':<30s} {'RMSE':>10s} {'MAE':>10s}")
        print(f"{'-'*60}")
        for method, result in comparison.items():
            print(f"{result['method_name']:<30s} "
                  f"{result['mean_rmse']:>10.4f} "
                  f"{result['mean_mae']:>10.4f}")
        
        return comparison


def main():
    """主评估函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Flexible Reference Forecast Evaluation')
    parser.add_argument('--data-dir', type=str, default='/datadir',
                       help='Data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of workers')
    parser.add_argument('--data-type', type=str, default='pangu',
                       choices=['pangu', 'era5_1deg', 'era5_025deg', 'custom'],
                       help='Data type')
    parser.add_argument('--n-channels', type=int, default=69,
                       help='Number of channels (for custom data type)')
    parser.add_argument('--n-lat', type=int, default=721,
                       help='Number of latitude grids (for custom data type)')
    parser.add_argument('--n-lon', type=int, default=1440,
                       help='Number of longitude grids (for custom data type)')
    
    args = parser.parse_args()
    
    # 创建数据配置
    if args.data_type == 'pangu':
        data_config = DataConfig.from_pangu()
    elif args.data_type == 'era5_1deg':
        data_config = DataConfig.from_era5_1deg(n_channels=args.n_channels)
    elif args.data_type == 'era5_025deg':
        data_config = DataConfig.from_era5_025deg(n_channels=args.n_channels)
    else:  # custom
        data_config = DataConfig(
            n_channels=args.n_channels,
            n_lat=args.n_lat,
            n_lon=args.n_lon
        )
    
    print(f"Using data configuration: {args.data_type}")
    print(f"  C={data_config.n_channels}, H={data_config.n_lat}, W={data_config.n_lon}")
    print(f"  Device: {args.device}")
    
    # 初始化评估器
    evaluator = FlexibleReferenceForecastEvaluator(
        data_config=data_config,
        data_dir=args.data_dir,
        device=args.device
    )
    
    # 加载验证数据
    print("\nLoading validation dataset...")
    from src.dataset.dataset_pangu import Dataset_pangu
    val_dataset = Dataset_pangu(split='val')
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # 运行评估
    results = evaluator.compare_all_methods(val_loader)
    
    # 保存结果
    save_path = f'reference_forecast_results_{args.data_type}.npy'
    np.save(save_path, results)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()