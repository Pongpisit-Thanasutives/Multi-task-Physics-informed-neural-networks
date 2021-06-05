import os
import glob
import numpy as np
from scipy.io import loadmat
from utils import pickle_load

def delta(arr):
    assert len(arr.shape)==1
    return arr[1]-arr[0]

def to_column_vector(arr):
    return arr.flatten()[:, None]

def load_indiv(data_path, real_solution=False, uniform=True, x_limit=None, t_limit=None):
    try:
        data = loadmat(data_path)
    except ValueError:
        data = pickle_load(data_path)

    time_key = 't' if 't' in data.keys() else 'tt'
    u_key = 'usol' if 'usol' in data.keys() else None
    if u_key is None:
        if 'u' in data.keys(): u_key = 'u'
        else: u_key = 'uu'
    spatial_key = 'x' if 'x' in data.keys() else 'xx'

    t = data[time_key].flatten()[:, None]
    x = data[spatial_key].flatten()[:, None]
    Exact = data[u_key]

    if uniform:
        print("Data is arranged in an uniform grid")
        if x_limit is not None: x = x[:x_limit, :]; Exact = Exact[:x_limit, :]
        if t_limit is not None: t = t[:t_limit, :]; Exact = Exact[:, :t_limit]
    else:
        print("Enforcing non-uniform grid")
        if x_limit is not None:
            idx_x = np.random.choice(len(x), x_limit, replace=False) 
            x = x[idx_x, :]; Exact = Exact[idx_x, :]
        if t_limit is not None:
            idx_t = np.random.choice(len(t), t_limit, replace=False)
            t = t[idx_t, :]; Exact = Exact[:, idx_t]

    if real_solution: Exact = np.real(Exact)
    
    return x, t, Exact.T

def space_time_grid(data_path, real_solution=False, uniform=True, x_limit=None, t_limit=None):
    try:
        data = loadmat(data_path)
    except ValueError:
        data = pickle_load(data_path)
    
    time_key = 't' if 't' in data.keys() else 'tt'
    u_key = 'usol' if 'usol' in data.keys() else None
    if u_key is None:
        if 'u' in data.keys(): u_key = 'u'
        else: u_key = 'uu'
    spatial_key = 'x' if 'x' in data.keys() else 'xx'
    
    t = data[time_key].flatten()[:, None]
    x = data[spatial_key].flatten()[:, None]
    Exact = data[u_key]

    if uniform:
        print("Data is arranged in an uniform grid")
        if x_limit is not None: x = x[:x_limit, :]; Exact = Exact[:x_limit, :]
        if t_limit is not None: t = t[:t_limit, :]; Exact = Exact[:, :t_limit]
    else:
        print("Enforcing non-uniform grid")
        if x_limit is not None:
            idx_x = np.random.choice(len(x), x_limit, replace=False) 
            x = x[idx_x, :]; Exact = Exact[idx_x, :]
        if t_limit is not None:
            idx_t = np.random.choice(len(t), t_limit, replace=False)
            t = t[idx_t, :]; Exact = Exact[:, idx_t]

    if real_solution: Exact = np.real(Exact)

    Exact = Exact.T
    X, T = np.meshgrid(x, t)
    
    return X, T, Exact

def get_trainable_data(X, T, Exact):
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]
    return X_star, u_star
