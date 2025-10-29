# AIWx - Persistence Forecast
"""
Persistence（持续性预报）

持续性预报假设天气状态保持不变，是最简单的参考预报方法。
在短期预报（6-24小时）中往往有较好的表现，但随着预报时效增加，
性能快速下降。

原理：
    将当前观测直接作为未来所有时刻的预报值。
    forecast(t+Δt) = observation(t), for all Δt

特点：
    - 无需额外数据或训练
    - 计算速度极快
    - 适合作为短期预报的性能下限
    - 在天气系统变化缓慢时表现较好
"""

import torch
from typing import Tuple, Optional


class Persistence:
    """
    持续性预报方法
    
    假设天气状态保持不变，将当前状态重复作为所有未来时刻的预报。
    """
    
    def __init__(self):
        """初始化持续性预报方法"""
        self.name = "Persistence"
        self.description = "Assumes weather state remains unchanged"
    
    def __call__(self, 
                 input_data: Tuple[torch.Tensor, torch.Tensor],
                 n_steps: int = 20) -> torch.Tensor:
        """
        生成持续性预报
        
        Args:
            input_data: 输入数据元组 (input_air, input_surface)
                - input_air: (batch, 5, 13, 721, 1440) 
                  5个大气变量 × 13个气压层
                - input_surface: (batch, 4, 721, 1440)
                  4个地表变量
            n_steps: 预报步数，默认20（对应120小时/5天）
        
        Returns:
            predictions: (batch, n_steps, 69, 721, 1440)
                所有时间步使用相同的初始状态
        
        Examples:
            >>> persistence = Persistence()
            >>> predictions = persistence((input_air, input_surface), n_steps=20)
            >>> print(predictions.shape)
            torch.Size([2, 20, 69, 721, 1440])
        """
        input_air, input_surface = input_data
        batch_size = input_air.shape[0]
        
        # 将输入重塑为目标格式
        # input_air: (batch, 5, 13, 721, 1440) -> (batch, 65, 721, 1440)
        air_flat = input_air.reshape(batch_size, 65, 721, 1440)
        
        # 拼接大气和地表变量: (batch, 69, 721, 1440)
        current_state = torch.cat([air_flat, input_surface], dim=1)
        
        # 重复n_steps次: (batch, n_steps, 69, 721, 1440)
        predictions = current_state.unsqueeze(1).repeat(1, n_steps, 1, 1, 1)
        
        return predictions
    
    def predict(self,
                input_data: Tuple[torch.Tensor, torch.Tensor],
                n_steps: int = 20) -> torch.Tensor:
        """
        predict方法的别名，与__call__功能相同
        
        提供此方法以保持接口一致性
        """
        return self(input_data, n_steps)
    
    def __repr__(self) -> str:
        return f"Persistence(name='{self.name}')"


if __name__ == "__main__":
    """测试代码"""
    print("="*60)
    print("Testing Persistence Forecast")
    print("="*60)
    
    # 创建测试数据
    batch_size = 2
    input_air = torch.randn(batch_size, 5, 13, 721, 1440)
    input_surface = torch.randn(batch_size, 4, 721, 1440)
    
    print(f"\nInput shapes:")
    print(f"  Air: {input_air.shape}")
    print(f"  Surface: {input_surface.shape}")
    
    # 初始化并预测
    persistence = Persistence()
    predictions = persistence((input_air, input_surface), n_steps=20)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Expected: ({batch_size}, 20, 69, 721, 1440)")
    
    # 验证：所有时间步应该相同
    assert predictions.shape == (batch_size, 20, 69, 721, 1440), "Shape mismatch!"
    
    first_step = predictions[:, 0, :, :, :]
    all_same = all(torch.allclose(predictions[:, t, :, :, :], first_step) 
                   for t in range(1, 20))
    
    if all_same:
        print("✓ All time steps are identical (as expected)")
    else:
        print("✗ Time steps differ (unexpected!)")
    
    # 验证数据来自输入
    input_combined = torch.cat([
        input_air.reshape(batch_size, 65, 721, 1440),
        input_surface
    ], dim=1)
    
    if torch.allclose(first_step, input_combined):
        print("✓ Predictions match input data")
    else:
        print("✗ Predictions don't match input")
    
    print("\n" + "="*60)
    print("✅ Persistence test completed successfully!")
    print("="*60)