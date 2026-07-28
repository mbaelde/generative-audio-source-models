import os
import numpy as np
import scipy.io as sio
from scipy.cluster.hierarchy import fcluster

def create_reduced_dictionary(feature_training, N_sounds, database_folder, fold, type_str, data_type):
    """
    Translated from create_reduced_dictionary.m
    """
    n_class = len(np.unique(feature_training[:, -2]))
    if np.isscalar(N_sounds) or len(np.atleast_1d(N_sounds)) == 1:
        N_sounds = np.ones(n_class) * np.atleast_1d(N_sounds)[0]
        
    N_spect = feature_training.shape[1] - 2
    n_cluster = np.zeros(n_class, dtype=int)
    idx_clusters = []
    
    for k in range(1, n_class + 1):
        feature_class = feature_training[feature_training[:, -2] == k, :]
        n_cluster[k-1] = int(np.floor(feature_class.shape[0] / N_sounds[k-1]))
        
        Z = None
        if type_str == 'single':
            if fold is None:
                path = f'Results/Dictionary reduction/{database_folder}Metaclasse/Z_{k}.mat'
            else:
                path = f'Clusters/{database_folder}Metaclasse/Fold {fold}/Z_{k}.mat'
            Z = sio.loadmat(path)['Z']
        elif type_str == 'mixture':
            if fold is None:
                path = f'Results/Dictionary reduction/{database_folder}Mixtures/Z_{k}.mat'
            else:
                path = f'Results/Mixture/Hierarchical/{database_folder}Mixtures/Fold {fold}/Z_{k}.mat'
            Z = sio.loadmat(path)['Z']
            
        idx = fcluster(Z, n_cluster[k-1], criterion='maxclust')
        idx_clusters.append(idx)
        print(f"Processed cluster {k}/{n_class}")
        
    maxclust = np.sum(n_cluster)
    my_feature = np.zeros((maxclust, N_spect + 2))
    cnt = 0
    for k in range(1, n_class + 1):
        feature_class = feature_training[feature_training[:, -2] == k, :]
        for nn in range(1, n_cluster[k-1] + 1):
            mean_feat = np.mean(feature_class[idx_clusters[k-1] == nn, :-2], axis=0)
            my_feature[cnt, :-2] = mean_feat
            my_feature[cnt, -2] = k
            my_feature[cnt, -1] = 1
            cnt += 1
            
    if data_type == 'raw':
        feature_norm = np.hstack([my_feature[:, :-2], my_feature[:, -2:]])
    elif data_type == 'energy':
        feature_norm = my_feature[:, :-2]
    elif data_type == 'feature':
        norm_factor = np.sum(my_feature[:, :-2], axis=1, keepdims=True)
        norm_factor[norm_factor == 0] = 1e-10
        feature_norm = np.hstack([my_feature[:, :-2] / norm_factor, my_feature[:, -2:]])
        
    return feature_norm
