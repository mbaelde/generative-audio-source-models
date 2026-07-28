from scipy.cluster.hierarchy import linkage

def my_linkage(Y, method='ward'):
    """
    Wrapper for hierarchical clustering linkage.
    In MATLAB, a custom fallback was used, but scipy handles linkages efficiently.
    Translated from my_linkage.m.
    """
    return linkage(Y, method=method)
