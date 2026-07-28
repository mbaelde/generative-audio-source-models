import numpy as np
import time

def identification_general(database, aux_L, prior_g, my_class, param):
    """
    Translated from identification_general.m
    """
    N_spect = param['N_spect']
    gpuFlag = param.get('gpuFlag', 0)
    n_buff = param.get('n_buff', 1)
    is_dict = param.get('dict', 1)
    
    N = database.shape[0]
    dict_size = aux_L.shape[0]
    
    database = database.T
    aux_L_comp = aux_L[:, :N_spect].T
    
    n_class = len(my_class)
    model_size = np.zeros(n_class)
    for k in range(n_class):
        model_size[k] = np.sum(aux_L[:, -2] == (k + 1))
        
    cum_model_size = np.cumsum(model_size).astype(int)
    msize = np.concatenate(([0], cum_model_size))
    
    # Avoid log(0)
    prior_g = np.maximum(prior_g, 1e-10)
    prior_g = np.log(prior_g / np.sum(prior_g))
    
    L_bay = np.zeros(N)
    L = np.zeros(dict_size)
    likelihood_group = np.zeros(len(msize) - 1)
    L_prior_max = np.zeros(len(msize) - 1)
    posterior_g = np.zeros((N, n_class))
    computation_time = np.zeros(N)
    
    for b in range(N):
        start_time = time.time()
        
        if is_dict:
            spectrum = database[:-2, b]
        else:
            data = database[:-2, b]
            spectrum = np.abs(np.fft.fft(data)) ** 2
            
        sum_spec = np.sum(spectrum[:N_spect])
        if sum_spec > 0:
            spectrum_norm = N_spect * spectrum[:N_spect] / sum_spec
        else:
            spectrum_norm = spectrum[:N_spect]
            
        # Pointwise multiplication of matching shapes and sum across freq dimension
        rspectrum_norm = np.tile(spectrum_norm, (dict_size, 1)).T
        L = np.sum(rspectrum_norm * aux_L_comp, axis=0) 
        
        for ii in range(len(msize) - 1):
            A = L[msize[ii]:msize[ii+1]]
            if len(A) > 0:
                L_prior_max[ii] = np.max(A)
                likelihood_group[ii] = np.log(np.nansum(np.exp(A - L_prior_max[ii])))
            else:
                L_prior_max[ii] = -np.inf
                likelihood_group[ii] = -np.inf
                
        likelihood_group = likelihood_group + L_prior_max - np.log(np.maximum(model_size, 1e-10))
        
        A = likelihood_group + prior_g
        L_prior_max_g = np.max(A)
        norm_factor_g = L_prior_max_g + np.log(np.sum(np.exp(A - L_prior_max_g)))
        posterior_g[b, :] = -norm_factor_g + likelihood_group + prior_g
        
        computation_time[b] = time.time() - start_time
        
        if (b + 1) % n_buff == 0:
            sum_g = np.nansum(posterior_g[b - n_buff + 1: b + 1, :], axis=0)
            max_idx = np.where(sum_g == np.max(sum_g))[0][0]
            L_bay[b - n_buff + 1: b + 1] = my_class[max_idx]
            
    return L_bay, posterior_g, computation_time
