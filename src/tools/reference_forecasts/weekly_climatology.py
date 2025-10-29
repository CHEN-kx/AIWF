# AIWx - Weekly Climatology Forecast
"""
Weekly Climatology（周气候态预报）

周气候态预报按年中的周次（week of year）分别计算气候态，相比简单的
气候态，能够更好地捕捉季节性变化。

原理：
    将历史数据按周分组，分别计算每周的平均态：
    forecast(t+Δt, week_w) = mean(historical_observations_week_w)
    
    预报时根据预报目标时刻所属的周次，选择对应的周气候态。

特点：
    - 考虑了季节循环（seasonal cycle）
    - 比简单气候态更精确
    - 需要更多存储空间（53周 × 气候态大小）
    - 对中长期预报（3-14天）提供更好的参考

应用场景：
    - 评估模型是否学到了季节性特征
    - 作为考虑季节变化的性能基准
    - 分析模型在不同季节的表现差异
"""

import os
import numpy as np
import torch
import datetime
from typing import Tuple, Optional, Union
from tqdm import tqdm


class WeeklyClimatology:
    """
    周气候态预报方法
    
    按年中周次分别计算气候态，能够捕捉季节性变化。
    """
    
    def __init__(self,
                 data_dir: str = '/datadir',
                 years: range = range(1979, 2015),
                 precomputed_path: Optional[str] = None):
        """
        初始化周气候态预报方法
        
        Args:
            data_dir: 数据根目录
            years: 用于计算气候态的年份范围
            precomputed_path: 预计算周气候态文件路径
        
        Examples:
            # 从预计算文件加载
            >>> weekly_clim = WeeklyClimatology(
            ...     precomputed_path='weekly_climatology.npy'
            ... )
            
            # 或指定数据目录和年份
            >>> weekly_clim = WeeklyClimatology(
            ...     data_dir='/datadir',
            ...     years=range(1979, 2015)
            ... )
            >>> weekly_clim.compute(save_path='weekly_climatology.npy')
        """
        self.name = "Weekly Climatology"
        self.description = "Uses week-of-year specific climatology"
        self.data_dir = data_dir
        self.years = years
        self.weekly_climatology = None  # (53, 69, 721, 1440)
        
        if precomputed_path and os.path.exists(precomputed_path):
            self.load(precomputed_path)
    
    def _parse_filename_to_week(self, filename: str) -> int:
        """
        从文件名解析周次
        
        Args:
            filename: 文件名，期望格式为 YYYYMMDDHH.npy
        
        Returns:
            week: 周次 (0-52)
        
        Examples:
            >>> self._parse_filename_to_week('2018010100.npy')
            0  # 1月1日是第0周
            >>> self._parse_filename_to_week('2018070100.npy')
            26  # 7月1日约是第26周
        """
        try:
            # 移除.npy后缀
            datestr = filename.replace('.npy', '')
            
            # 解析日期 (YYYYMMDDHH)
            if len(datestr) >= 8:
                year = int(datestr[:4])
                month = int(datestr[4:6])
                day = int(datestr[6:8])
                
                # 创建日期对象
                date = datetime.date(year, month, day)
                
                # 获取ISO周次 (1-53)，转为0-52
                week = date.isocalendar()[1] - 1
                
                # 确保在有效范围内
                return max(0, min(52, week))
            else:
                # 无法解析，返回0
                return 0
                
        except Exception as e:
            print(f"Warning: Failed to parse date from {filename}: {e}")
            return 0
    
    def compute(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        从历史数据计算周气候态
        
        Args:
            save_path: 保存路径（可选）
        
        Returns:
            weekly_climatology: (53, 69, 721, 1440) 每周的气候态
        
        Note:
            此操作可能需要几小时，且需要较大内存（~100GB）。
            建议使用较少年份进行测试，确认无误后再使用完整数据。
        """
        print(f"\nComputing weekly climatology from years {self.years.start}-{self.years.stop-1}...")
        print(f"Data directory: {self.data_dir}")
        
        # 为每周创建数据列表
        weekly_data = {week: [] for week in range(53)}
        
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
                    # 解析周次
                    week = self._parse_filename_to_week(filename)
                    
                    # 加载数据
                    data = np.load(filepath)
                    
                    # 验证形状
                    if data.shape != (69, 721, 1440):
                        print(f"Warning: {filename} has unexpected shape {data.shape}")
                        continue
                    
                    # 添加到对应周的列表
                    weekly_data[week].append(data)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
        
        # 计算每周的平均
        print("\nComputing mean for each week...")
        self.weekly_climatology = np.zeros((53, 69, 721, 1440), dtype=np.float32)
        
        for week in tqdm(range(53), desc="Computing weekly means"):
            if len(weekly_data[week]) > 0:
                week_stack = np.stack(weekly_data[week], axis=0)
                self.weekly_climatology[week] = np.mean(week_stack, axis=0)
                
                if week % 10 == 0:  # 每10周打印一次统计
                    print(f"  Week {week:2d}: {len(weekly_data[week]):5d} samples")
            else:
                print(f"  Warning: Week {week} has no data!")
                # 使用前一周的数据作为替代
                if week > 0:
                    self.weekly_climatology[week] = self.weekly_climatology[week-1]
                    print(f"    Using week {week-1} data as fallback")
        
        print(f"\nWeekly climatology shape: {self.weekly_climatology.shape}")
        print(f"Memory usage: ~{self.weekly_climatology.nbytes / 1024**3:.2f} GB")
        
        # 保存
        if save_path:
            print(f"\nSaving to {save_path}...")
            np.save(save_path, self.weekly_climatology)
            print(f"✓ Weekly climatology saved")
        
        return self.weekly_climatology
    
    def load(self, path: str):
        """
        从文件加载预计算的周气候态
        
        Args:
            path: 周气候态文件路径（.npy格式）
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weekly climatology file not found: {path}")
        
        self.weekly_climatology = np.load(path)
        print(f"✓ Weekly climatology loaded from {path}")
        print(f"  Shape: {self.weekly_climatology.shape}")
        print(f"  Memory: ~{self.weekly_climatology.nbytes / 1024**3:.2f} GB")
        
        # 验证形状
        if self.weekly_climatology.shape != (53, 69, 721, 1440):
            raise ValueError(
                f"Invalid weekly climatology shape: {self.weekly_climatology.shape}"
            )
    
    def __call__(self,
                 input_data: Tuple[torch.Tensor, torch.Tensor],
                 n_steps: int = 20,
                 initial_week: int = 0) -> torch.Tensor:
        """
        生成周气候态预报
        
        Args:
            input_data: 输入数据（保持接口一致）
            n_steps: 预报步数
            initial_week: 初始时刻的周次 (0-52)
        
        Returns:
            predictions: (batch, n_steps, 69, 721, 1440)
                根据预报时效选择对应周的气候态
        
        Note:
            假设预报间隔为6小时，则：
            - 4步 = 1天
            - 28步 = 1周
            因此每28步会切换到下一周的气候态
        """
        if self.weekly_climatology is None:
            raise RuntimeError(
                "Weekly climatology not computed! Call compute() or load() first."
            )
        
        input_air, input_surface = input_data
        batch_size = input_air.shape[0]
        device = input_air.device
        
        # 转换为torch tensor
        weekly_clim_torch = torch.from_numpy(self.weekly_climatology).to(device)
        
        # 为每个预报步选择对应周次的气候态
        predictions = []
        
        for step in range(n_steps):
            # 计算该步对应的周次
            # 假设6小时一步，28步=1周
            week_offset = step // 28
            forecast_week = (initial_week + week_offset) % 53
            
            # 获取该周的气候态
            week_clim = weekly_clim_torch[forecast_week]  # (69, 721, 1440)
            predictions.append(week_clim)
        
        # 堆叠成 (n_steps, 69, 721, 1440)
        predictions = torch.stack(predictions, dim=0)
        
        # 扩展batch维度: (batch, n_steps, 69, 721, 1440)
        predictions = predictions.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        
        return predictions
    
    def predict(self,
                input_data: Tuple[torch.Tensor, torch.Tensor],
                n_steps: int = 20,
                initial_week: int = 0) -> torch.Tensor:
        """predict方法的别名"""
        return self(input_data, n_steps, initial_week)
    
    def __repr__(self) -> str:
        computed = self.weekly_climatology is not None
        return f"WeeklyClimatology(name='{self.name}', computed={computed})"


if __name__ == "__main__":
    """测试代码"""
    print("="*60)
    print("Testing Weekly Climatology Forecast")
    print("="*60)
    
    # 创建虚拟周气候态用于测试
    print("\n1. Creating dummy weekly climatology for testing...")
    fake_weekly_clim = np.random.randn(53, 69, 721, 1440).astype(np.float32)
    temp_path = '/tmp/test_weekly_climatology.npy'
    np.save(temp_path, fake_weekly_clim)
    print(f"   Saved to {temp_path}")
    print(f"   Size: ~{fake_weekly_clim.nbytes / 1024**3:.2f} GB")
    
    # 测试加载
    print("\n2. Testing load functionality...")
    weekly_clim = WeeklyClimatology(precomputed_path=temp_path)
    
    # 测试文件名解析
    print("\n3. Testing filename parsing...")
    test_cases = [
        ('2018010100.npy', 0),    # Jan 1
        ('2018040100.npy', 13),   # Apr 1, ~week 13
        ('2018070100.npy', 26),   # Jul 1, ~week 26
        ('2018100100.npy', 39),   # Oct 1, ~week 39
    ]
    
    for filename, expected_range in test_cases:
        week = weekly_clim._parse_filename_to_week(filename)
        print(f"   {filename} -> Week {week} (expected ~{expected_range})")
    
    # 测试预测
    print("\n4. Testing prediction...")
    batch_size = 2
    input_air = torch.randn(batch_size, 5, 13, 721, 1440)
    input_surface = torch.randn(batch_size, 4, 721, 1440)
    
    # 测试不同初始周次
    for initial_week in [0, 10, 26, 52]:
        predictions = weekly_clim(
            (input_air, input_surface),
            n_steps=20,
            initial_week=initial_week
        )
        print(f"   Initial week {initial_week:2d}: predictions shape {predictions.shape}")
        
        # 验证形状
        assert predictions.shape == (batch_size, 20, 69, 721, 1440)
    
    print("   ✓ All predictions have correct shape")
    
    # 验证：检查前几个时间步使用的周次
    print("\n5. Verifying week-specific climatology...")
    initial_week = 0
    predictions = weekly_clim((input_air, input_surface), n_steps=20, initial_week=0)
    
    for t in range(min(20, 3)):  # 检查前3步
        week_offset = t // 28
        expected_week = (initial_week + week_offset) % 53
        expected_clim = fake_weekly_clim[expected_week]
        
        pred_t = predictions[0, t, :, :, :].cpu().numpy()
        if np.allclose(pred_t, expected_clim):
            print(f"   Step {t}: Uses week {expected_week} ✓")
        else:
            print(f"   Step {t}: Week {expected_week} mismatch ✗")
    
    # 清理
    os.remove(temp_path)
    
    print("\n" + "="*60)
    print("✅ Weekly Climatology test completed successfully!")
    print("="*60)