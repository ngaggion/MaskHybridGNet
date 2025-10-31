import json
from pathlib import Path
import numpy as np
import torch
import scipy.sparse as sp

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

def load_config(dataset_path, hyperparameters = None):
    config = {}
    config['DATASET'] = dataset_path
    
    with open(Path(dataset_path) / "config.json") as f:
        data_config = json.load(f)

    if hyperparameters is None:
        hyperparameters = {}
        hyperparameters['latents'] = 64
        hyperparameters['initial_filters'] = 16
    
    # Data augmentation is left to the dataset construction info
    hyperparameters['flip_h'] = data_config['flip_h']
    hyperparameters['flip_v'] = data_config['flip_v']
    hyperparameters['transpose'] = data_config['transpose']
    hyperparameters['rotate'] = data_config['rotate'] 
    
    config.update({
        'organs': data_config['organs'],
        'organ_names': data_config['organ_names'],
        'resolutions': data_config['resolutions'],
        'inputsize': data_config['inputsize'],
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    })
    
    config.update(hyperparameters)

    num_pooling = len(config['resolutions'])
    config['filters'] = [config['initial_filters'] * (i+1) // 2 for i in range(num_pooling+1)]
    config['filters'] = [x for x in config['filters'] for _ in (0, 1)]
    config['filters'][0] = 2
    
    if config['naive']:
        adj_path = "Naive"
    else:
        adj_path = "NonNaive"
        
    print("Loading adjacency matrices", Path(dataset_path) / adj_path / f"adj_full_block_diagonal.npy")

    A_ = []
    for res in config['resolutions']:
        A = np.load(Path(dataset_path) / adj_path / f"adj_{res}_block_diagonal.npy")
        A = sp.csc_matrix(A).tocoo()
        A_.extend([A.copy()])
    A_.append(A_[-1])
    A_t = [scipy_to_torch_sparse(x).to(config['device']) for x in A_]
    config['n_nodes'] = [A.shape[0] for A in A_]

    D_ = []
    for res in ['to_' + x for x in config['resolutions'][1:]]:
        D = np.load(Path(dataset_path) / adj_path /  f"downsampling_{res}.npy")
        D_.append(sp.csc_matrix(D).tocoo())
    D_t = [scipy_to_torch_sparse(x).to(config['device']) for x in D_]

    U_ = []
    for res in ['to_' + x for x in config['resolutions'][:-1]]:
        U = np.load(Path(dataset_path) / adj_path /  f"upsampling_{res}.npy")
        U_.append(sp.csc_matrix(U).tocoo())
    U_t = [scipy_to_torch_sparse(x).to(config['device']) for x in U_]

    return config, D_t, U_t, A_t