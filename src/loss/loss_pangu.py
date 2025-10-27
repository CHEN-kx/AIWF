# AIWx
# We conducted the reproduction based on the pseudo-code provided by the official website 
# https://github.com/198808xc/Pangu-Weather
# <Accurate medium-range global weather forecasting with 3D neural networks>

import os
import sys
CURRENT_DIR = os.path.split(os.path.abspath(__file__))[0]  # 当前目录
config_path = CURRENT_DIR.rsplit('/', 2)[0]  # 上三级目录
sys.path.append(config_path)

import torch

from src.tools.metrics import create_latitude_weights

def latitude_and_variable_weighted_mae(y_true_upper, y_true_surface, y_pred_upper, y_pred_surface):
    weights = create_latitude_weights(721) # 721
    error_upper = torch.abs(y_pred_upper - y_true_upper).mean((0,2,4)) # 5, 721
    weights_upper = torch.from_numpy(np.tile(weights, (5, 1))).to(y_pred_upper.device)
    latitude_weighted_error_upper = (error_upper * weights_upper).mean(1) # 5
    var_weighted_error_upper = (latitude_weighted_error_upper*torch.tensor([2.5, 0.60, 1.25, 0.77, 0.54], device=y_pred_upper.device)).mean()

    error_surface = torch.abs(y_pred_surface - y_true_surface).mean((0,3)) # 5, 721
    weights_surface = torch.from_numpy(np.tile(weights, (5, 1))).to(device=y_pred_surface.device)
    latitude_weighted_error_surface = (error_surface * weights_surface).mean(1) # 5
    var_weighted_error_surface = (latitude_weighted_error_surface*torch.tensor([3.0, 0.8, 0.66, 2.75, 2.0], device=y_pred_surface.device)).mean()

    return var_weighted_error_upper + 0.25*var_weighted_error_surface