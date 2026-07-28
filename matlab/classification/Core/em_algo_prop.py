import numpy as np

def em_algo_prop(mixture_pdf, feature_norm, n_try, iter_max, tol, verbose=False):
    """
    EM algorithm for a given mixture model.
    Translated from em_algo_prop.m
    """
    M, N_spect = feature_norm.shape
    mixture_pdf = np.asarray(mixture_pdf).flatten()
    
    BIC = np.zeros(n_try)
    mprop_em = np.zeros((M, n_try))
    
    for ntry in range(n_try):
        if verbose:
            print(f'--- try: {ntry + 1} ---')
            
        prop_em = np.random.rand(M)
        prop_em /= np.sum(prop_em)
        
        L = np.zeros(iter_max)
        iter_cur = 1
        stop = False
        
        while not stop:
            # E-step
            expectation = prop_em[:, None] * feature_norm
            expectation /= np.maximum(np.sum(expectation, axis=0), 1e-10)
            
            # M-step
            prop_em = np.sum(mixture_pdf[None, :] * expectation, axis=1) / np.maximum(np.sum(mixture_pdf), 1e-10)
            
            # Likelihood
            inner_sum = np.sum(prop_em[:, None] * feature_norm, axis=0)
            L[iter_cur - 1] = np.sum(mixture_pdf * np.log(np.maximum(inner_sum, 1e-10)))
            
            if verbose:
                if iter_cur == 1:
                    print(f'iter: {iter_cur:2d} ; L: {L[iter_cur-1]:5.4f} ; diff : inf')
                else:
                    diff = L[iter_cur-1] - L[iter_cur-2]
                    print(f'iter: {iter_cur:2d} ; L: {L[iter_cur-1]:5.4f} ; diff : {diff:1.6f}')
            
            iter_cur += 1
            if iter_cur == 2:
                stop = (iter_cur > iter_max) or (abs(L[iter_cur-2]) < tol)
            else:
                stop = (iter_cur > iter_max) or (abs(L[iter_cur-2] - L[iter_cur-3]) < tol)
                
        BIC[ntry] = -2 * L[iter_max - 1] + M * np.log(N_spect)
        if verbose:
            print(f'BIC:  {BIC[ntry]:5.4f}')
        mprop_em[:, ntry] = prop_em
        
    best_idx = np.nanargmin(BIC)
    return mprop_em[:, best_idx]
