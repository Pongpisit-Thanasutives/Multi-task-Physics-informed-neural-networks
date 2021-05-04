import os
import glob
import numpy as np
from scipy.io import loadmat

def space_time_grid(data_path, real_solution=False):
    data = loadmat(data_path)
    
    time_key = 't' if 't' in data.keys() else 'tt'
    u_key = 'usol' if 'usol' in data.keys() else None
    if u_key is None:
        if 'u' in data.keys(): u_key = 'u'
        else: u_key = 'uu'
    spatial_key = 'x' if 'x' in data.keys() else 'xx'
    
    t = data[time_key].flatten()[:, None]
    x = data[spatial_key].flatten()[:, None]
    Exact = data[u_key]
    if real_solution: Exact = np.real(Exact)
    Exact = Exact.T
    X, T = np.meshgrid(x, t)
    
    return X, T, Exact

def get_trainable_data(X, T, Exact):
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    return X_star, u_star
