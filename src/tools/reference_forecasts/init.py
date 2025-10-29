# AIWx - Reference Forecast Methods
"""
参考预报方法（Reference Forecast Methods）

本模块提供多种参考预报方法，用于评估AI气象模型的性能基准。
这些方法不是真正的"baseline"模型，而是用于对比的参考标准。

包含的方法：
- Persistence: 持续性预报（假设天气状态不变）
- Climatology: 气候态预报（使用历史平均）
- WeeklyClimatology: 周气候态预报（考虑季节变化）

使用示例：
    from src.tools.reference_forecasts import Persistence
    
    persistence = Persistence()
    predictions = persistence(inputs, n_steps=20)
"""

from .persistence import Persistence
from .climatology import Climatology
from .weekly_climatology import WeeklyClimatology

__all__ = ['Persistence', 'Climatology', 'WeeklyClimatology']

__version__ = '1.0.0'