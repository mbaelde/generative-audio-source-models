import numpy as np

def estimate_mixtspectrum(feature_class, spectrum_test):
    """
    EM algorithm for estimating mixture spectrum.
    Translated from estimate_mixtspectrum.m
    """
    M, N = feature_class.shape
    spectrum_test = np.asarray(spectrum_test).flatten()
    
    prop_em = np.ones(M) / M
    iter_max = 30
    likelihood = np.zeros(iter_max + 1)
    likelihood[0] = np.inf
    
    z = None
    for iteration in range(iter_max):
        # E-step
        z = prop_em[:, None] * feature_class
        z /= np.maximum(np.sum(z, axis=0), 1e-10)
        
        # M-step
        prop_em = np.sum(z * spectrum_test[None, :], axis=1) / np.maximum(np.sum(spectrum_test), 1e-10)
        
        # Likelihood
        inner_sum = np.sum(prop_em[:, None] * feature_class, axis=0)
        likelihood[iteration + 1] = np.sum(np.log(np.maximum(inner_sum, 1e-10)))
        
        if iteration > 0 and abs((likelihood[iteration + 1] - likelihood[iteration]) / max(abs(likelihood[iteration + 1]), 1e-10)) < 1e-5:
            likelihood = likelihood[:iteration + 2]
            break
            
    est_spectrum = np.sum(prop_em[:, None] * feature_class, axis=0)
    return est_spectrum, prop_em, likelihood, z
