# AIWx - Climatology Forecast
"""
Climatology（气候态预报）

气候态预报使用历史多年平均值作为预报结果。这种方法假设天气遵循
长期气候平均状态，忽略了具体天气系统的演变。

原理：
    使用历史数据的时间平均作为预报：
    forecast(t+Δt) = mean(historical_observations), for all t, Δt

特点：
    - 需要预先计算历史数据的平均态
    - 对长期预报（>7天）提供有意义的参考
    - 能够捕捉大尺度的气候特征
    - 忽略了短期天气系统的变化

应用场景：
    - 长期预报（7-14天）的性能基准
    - 评估模型是否学到了气候态信息
    - 作为异常检测的参考标准
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional, Union
from tqdm import tqdm


class Climatology:
    """
    气候态预报方法
    
    使用历史多年平均值作为所有时刻的预报。
    """
    
    def __init__(self,
                 data_dir: str = '/datadir',
                 years: range = range(1979, 2015),
                 precomputed_path: Optional[str] = None):
        """
        初始化气候态预报方法
        
        Args:
            data_dir: 数据根目录
            years: 用于计算气候态的年份范围（默认1979-2014）
            precomputed_path: 预计算气候态文件路径（如存在则直接加载）
        
        Examples:
            # 从预计算文件加载
            >>> clim = Climatology(precomputed_path='climatology.npy')
            
            # 或指定数据目录和年份
            >>> clim = Climatology(data_dir='/datadir', years=range(1979, 2015))
            >>> clim.compute(save_path='climatology.npy')  # 计算并保存
        """
        self.name = "Climatology"
        self.description = "Uses multi-year mean as forecast"
        self.data_dir = data_dir
        self.years = years
        self.climatology = None
        
        # 如果提供了预计算文件，尝试加载
        if precomputed_path and os.path.exists(precomputed_path):
            self.load(precomputed_path)
    
    def compute(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        从历史数据计算气候态
        
        Args:
            save_path: 保存路径（可选）
        
        Returns:
            climatology: (69, 721, 1440) 气候平均态
        
        Note:
            此操作可能需要几小时，取决于数据量和IO速度。
            建议首次计算后保存结果，后续直接加载。
        """
        print(f"\nComputing climatology from years {self.years.start}-{self.years.stop-1}...")
        print(f"Data directory: {self.data_dir}")
        
        all_data = []
        total_files = 0
        
        # 遍历所有年份
        for year in tqdm(self.years, desc="Processing years"):
            year_dir = os.path.join(self.data_dir, str(year))
            
            if not os.path.exists(year_dir):
                print(f"Warning: {year_dir} not found, skipping...")
                continue
            
            # 加载该年份的所有文件
            files = sorted([f for f in os.listdir(year_dir) if f.endswith('.npy')])
            
            for filename in files:
                filepath = os.path.join(year_dir, filename)
                try:
                    data = np.load(filepath)  # 期望形状: (69, 721, 1440)
                    
                    # 验证形状
                    if data.shape != (69, 721, 1440):
                        print(f"Warning: {filename} has unexpected shape {data.shape}, skipping...")
                        continue
                    
                    all_data.append(data)
                    total_files += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
        
        if len(all_data) == 0:
            raise ValueError(f"No valid data found in {self.data_dir}!")
        
        print(f"\nLoaded {total_files} files from {len(self.years)} years")
        
        # 计算平均
        print("Computing mean across all samples...")
        all_data = np.stack(all_data, axis=0)  # (n_samples, 69, 721, 1440)
        self.climatology = np.mean(all_data, axis=0).astype(np.float32)  # (69, 721, 1440)
        
        print(f"Climatology shape: {self.climatology.shape}")
        print(f"Memory usage: ~{self.climatology.nbytes / 1024**3:.2f} GB")
        
        # 保存
        if save_path:
            np.save(save_path, self.climatology)
            print(f"✓ Climatology saved to {save_path}")
        
        return self.climatology
    
    def load(self, path: str):
        """
        从文件加载预计算的气候态
        
        Args:
            path: 气候态文件路径（.npy格式）
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Climatology file not found: {path}")
        
        self.climatology = np.load(path)
        print(f"✓ Climatology loaded from {path}")
        print(f"  Shape: {self.climatology.shape}")
        print(f"  Memory: ~{self.climatology.nbytes / 1024**3:.2f} GB")
        
        # 验证形状
        if self.climatology.shape != (69, 721, 1440):
            raise ValueError(f"Invalid climatology shape: {self.climatology.shape}")
    
    def __call__(self,
                 input_data: Tuple[torch.Tensor, torch.Tensor],
                 n_steps: int = 20) -> torch.Tensor:
        """
        生成气候态预报
        
        Args:
            input_data: 输入数据（保持接口一致，但不使用）
            n_steps: 预报步数
        
        Returns:
            predictions: (batch, n_steps, 69, 721, 1440)
                所有时间步使用相同的气候态
        """
        if self.climatology is None:
            raise RuntimeError(
                "Climatology not computed! Call compute() or load() first."
            )
        
        input_air, input_surface = input_data
        batch_size = input_air.shape[0]
        device = input_air.device
        
        # 转换为torch tensor并移到相应设备
        clim_torch = torch.from_numpy(self.climatology).to(device)
        
        # 扩展到batch和时间维度
        predictions = clim_torch.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, n_steps, 1, 1, 1
        )
        
        return predictions
    
    def predict(self,
                input_data: Tuple[torch.Tensor, torch.Tensor],
                n_steps: int = 20) -> torch.Tensor:
        """predict方法的别名"""
        return self(input_data, n_steps)
    
    def __repr__(self) -> str:
        computed = self.climatology is not None
        return f"Climatology(name='{self.name}', computed={computed})"


if __name__ == "__main__":
    """测试代码"""
    print("="*60)
    print("Testing Climatology Forecast")
    print("="*60)
    
    # 创建虚拟气候态用于测试
    print("\n1. Creating dummy climatology for testing...")
    fake_climatology = np.random.randn(69, 721, 1440).astype(np.float32)
    temp_path = '/tmp/test_climatology.npy'
    np.save(temp_path, fake_climatology)
    print(f"   Saved to {temp_path}")
    
    # 测试加载
    print("\n2. Testing load functionality...")
    clim = Climatology(precomputed_path=temp_path)
    print(f"   Loaded climatology shape: {clim.climatology.shape}")
    
    # 测试预测
    print("\n3. Testing prediction...")
    batch_size = 2
    input_air = torch.randn(batch_size, 5, 13, 721, 1440)
    input_surface = torch.randn(batch_size, 4, 721, 1440)
    
    predictions = clim((input_air, input_surface), n_steps=20)
    print(f"   Predictions shape: {predictions.shape}")
    
    # 验证
    assert predictions.shape == (batch_size, 20, 69, 721, 1440), "Shape mismatch!"
    print("   ✓ Shape correct")
    
    # 验证所有批次和时间步使用相同的气候态
    for b in range(batch_size):
        for t in range(20):
            pred_bt = predictions[b, t, :, :, :].cpu().numpy()
            if not np.allclose(pred_bt, fake_climatology):
                raise AssertionError(f"Batch {b}, time {t} mismatch!")
    
    print("   ✓ All predictions use the same climatology")
    
    # 清理
    os.remove(temp_path)
    
    print("\n" + "="*60)
    print("✅ Climatology test completed successfully!")
    print("="*60)