# AIWF: AI for Weather Forecasting Codebase

[![AIWx](https://img.shields.io/badge/Gropu-AIWx-009688.svg)](https://github.com/198808xc/Pangu-Weather)  ![AIWx Research](https://img.shields.io/badge/Research_Focus-AI4Weather-0077b6.svg) [![Academic Partnership](https://img.shields.io/badge/Academic_Collaboration-Open-8a2be2.svg)](https://aiwx.org/collaborate)

## Framework Value Proposition 
This codebase systematically integrates and reimplements state-of-the-art meteorological AI models to advance operational workflows in weather forecasting. Developed by AIWx, the framework provides:

1. **Standardized Benchmarking**  
   Unified evaluation protocols across multiple models (RMSE, ACC, CRPS)  
   Cross-modal validation datasets (ERA5, GFS, CMIP6)

2. **Production-ready Components**  
   Hybrid architecture supporting both research experiments and operational systems  
   Automated pipeline from data preprocessing to ensemble forecasting

3. **Community-driven Development**  
   Detailed reproducibility guidelines

## Implemented Models:  
- [Pangu-Weather](https://github.com/198808xc/Pangu-Weather) - Accurate medium-range global weather forecasting with 3D neural networks

## Changelog

### [2025-10-27]

#### Added
- Core infrastructure scaffolding ([@kaixin](https://github.com/CHEN-kx))
  - Modular architecture pattern: `dataset/`, `loss/`, `model/`, `tools/`, `trainer/`

#### Features
- Pangu-Weather implementation modules ([@kaixin](https://github.com/CHEN-kx))
  - Model architecture: 3D Earth-specific transformer blocks
  - Training framework: Weighted RMSE loss function

### [2025-10-28]

#### Added
- Reference forecast evaluation system
  - `tools/reference_forecasts/`: Modular baseline forecasting methods
    - Persistence, Climatology, Weekly Climatology
  - `tools/improved_metrics.py`: Flexible metrics with configurable data dimensions
  - Comprehensive evaluation pipeline for model benchmarking

## Reference Forecasts

The `reference_forecasts` module provides standard baseline methods for evaluating AI weather models. These methods quantify model skill improvement over conventional forecasting approaches.

**Available Methods:**
- **Persistence**: Assumes unchanged weather state (baseline for short-range forecasts)
- **Climatology**: Historical mean-based forecasting (baseline for long-range)
- **Weekly Climatology**: Seasonal-aware climatological forecasts

**Key Features:**
- Configurable data dimensions (C, H, W) for different datasets
- Latitude-weighted evaluation metrics (RMSE, MAE, ACC, Bias)
- Automated comparison against AI model predictions

**Quick Example:**
```python
from src.tools.reference_forecasts import Persistence
from src.tools.improved_metrics import WeatherMetrics, DataConfig

config = DataConfig.from_pangu()  # C=69, H=721, W=1440
persistence = Persistence()
metrics = WeatherMetrics(config)

predictions = persistence((input_air, input_surface), n_steps=20)
rmse = metrics.compute_rmse(predictions, targets)
```

## Technical References & Acknowledgments
Core architecture derived from the pioneering work of:  
- [Pangu-Weather Official Implementation](https://github.com/198808xc/Pangu-Weather)

## Contact us
For technical inquiries or collaborative research opportunities in AI-driven meteorology, please direct your correspondence to:
- Kaixin Chen (chenkaixin@bupt.edu.cn | Beijing University of Posts and Telecommunications)

