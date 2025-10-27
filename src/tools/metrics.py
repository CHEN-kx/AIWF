# AIWx

import numpy as np

import torch

def create_latitude_weights(shape, lat_start=90.0, lat_end=-90.0):
    latitudes = np.linspace(lat_start, lat_end, shape)
    weights = np.cos(np.deg2rad(latitudes))
    weights /= weights.mean()
    return weights

def compute_latitude_weighted_rmse(self, out, tgt):
    '''Batch Sum latitude_weighted_RMSE'''
    weights = torch.from_numpy(create_latitude_weights(721)).to(out.device)
    error = ((out - tgt)**2).mean(2)
    error = torch.sqrt((error*weights).mean(1))
    return error.sum()