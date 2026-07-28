import numpy as np
from scipy.stats import multivariate_normal, beta, dirichlet
from scipy.special import digamma
from scipy.optimize import newton

def invpsi(y):
    """
    Inverse of the digamma function.
    """
    y = np.asarray(y)
    out = np.empty_like(y)
    
    for i, val in np.ndenumerate(y):
        if val >= -2.22:
            x0 = np.exp(val) + 0.5
        else:
            x0 = -1.0 / (val - digamma(1))
        
        try:
            out[i] = newton(lambda x: digamma(x) - val, x0)
        except RuntimeError:
            out[i] = x0 # fallback
    return out if out.shape else out.item()


def logmvnpdf(x, mu, cov):
    """
    Log likelihood array for observations x where x_n ~ N(mu, cov)
    x is (N, D), mu is (D,), cov is (D, D)
    """
    return multivariate_normal.logpdf(x, mean=mu, cov=cov)

def mvbetapdf(x, a, b):
    """
    Product of univariate beta PDFs
    x is (N, L), a is (L,), b is (L,)
    """
    out = np.ones(x.shape[0])
    for l in range(len(a)):
        out *= beta.pdf(x[:, l], a[l], b[l])
    return out

def mixture_dirichlet_fit(r, M, iter_max):
    """
    EM algorithm to fit a Dirichlet mixture.
    r is (N, K)
    M is number of mixtures
    """
    N, K = r.shape
    
    prop = np.ones(M) / M
    a = 10 * np.random.rand(M, K)
    
    L = np.zeros(iter_max)
    
    for iteration in range(iter_max):
        a_old = a.copy()
        prop_old = prop.copy()
        z = np.zeros((M, N))
        
        # E-step
        for m in range(M):
            # Evaluate dirichlet PDF for each observation
            # scipy.stats.dirichlet requires sum(x) == 1 for each row
            # We assume r is properly normalized.
            # Handle potential zeros or NaNs by adding a tiny epsilon if needed, 
            # but we assume r is well-formed here.
            try:
                pdfs = dirichlet.pdf(r.T, a_old[m, :])
            except ValueError:
                # If a is invalid or r is not simplex
                pdfs = np.zeros(N)
                
            z[m, :] = prop_old[m] * pdfs
            
        z = z / np.maximum(np.sum(z, axis=0), 1e-10) # avoid div by zero
        
        # M-step
        prop = np.sum(z, axis=1) / N
        
        for m in range(M):
            for k in range(K):
                # Update alpha parameters using invpsi
                num = np.sum(z[m, :] * digamma(np.sum(a_old[m, :])) + np.log(r[:, k]))
                den = np.sum(z[m, :])
                if den > 1e-10:
                    a[m, k] = invpsi(num / den)
                
        # Likelihood
        aux = np.zeros(N)
        for m in range(M):
            try:
                pdfs = dirichlet.pdf(r.T, a[m, :])
            except ValueError:
                pdfs = np.zeros(N)
            aux += prop[m] * pdfs
            
        # Avoid log(0)
        aux = np.maximum(aux, 1e-10)
        L[iteration] = np.sum(np.log(aux))
        
    return a, prop, L
