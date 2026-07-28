import os
import numpy as np
import scipy.io as sio
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

def cluster_per_class(data_folder, folder_database, dict_idx, fold, fs_list, T_list, class_names):
    """
    Clusters features per class using Ward's linkage.
    Translated from cluster_per_class.m.
    """
    fs = fs_list[dict_idx]
    T = T_list[dict_idx]
    
    base_path = os.path.join('Database', f'{folder_database}FS{fs}', f'T{T}', 'Metaclasse')
    file_path = os.path.join(base_path, f'dataset_T{T}_fold_{fold}.mat')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    data = sio.loadmat(file_path)
    feature_training = data['feature_training']
    n_class = len(class_names)
    
    out_dir = os.path.join('Clusters', f'{folder_database}Uniform', f'Fold {fold}')
    os.makedirs(out_dir, exist_ok=True)
    
    for k in range(2, n_class + 1):
        # In MATLAB, labels are usually 1-indexed. Assuming the label is in the second to last column.
        feature_class = feature_training[feature_training[:, -2] == k, :]
        
        # Normalize
        feats = feature_class[:, :-2]
        feats = feats / np.maximum(np.sum(feats, axis=1, keepdims=True), 1e-10)
        
        # Clustering
        Y = pdist(np.sqrt(feats / 2.0))
        Z = linkage(Y, method='ward')
        
        # Save
        out_file = os.path.join(out_dir, f'Z_{k}.mat')
        sio.savemat(out_file, {'Z': Z})
        print(f"Processed class {k}/{n_class}")
