# AIWF: AI for Weather Forecasting Codebase

[![AIWx](https://img.shields.io/badge/Gropu-AIWx-009688.svg)](https://github.com/198808xc/Pangu-Weather)  ![AIWx Research](https://img.shields.io/badge/Research_Focus-AI4Weather-0077b6.svg)

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

## Technical References & Acknowledgments
Core architecture derived from the pioneering work of:  
- [Pangu-Weather Official Implementation](https://github.com/198808xc/Pangu-Weather)
