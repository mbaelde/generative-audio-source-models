import numpy as np

def metrics_sed(matrix_true, matrix_pred):
    """
    Translated from metrics_sed.m
    """
    n_class, N = matrix_true.shape
    
    true_positive = np.zeros(N)
    false_positive = np.zeros(N)
    false_negative = np.zeros(N)
    substitution = np.zeros(N)
    deletion = np.zeros(N)
    insertion = np.zeros(N)
    n_active = np.zeros(N)
    
    for n in range(N):
        true_col = matrix_true[:, n]
        pred_col = matrix_pred[:, n]
        
        for k in range(n_class):
            if true_col[k] == 1 and pred_col[k] == 1:
                true_positive[n] += 1
            elif true_col[k] == 1 and pred_col[k] == 0:
                false_negative[n] += 1
            elif true_col[k] == 0 and pred_col[k] == 1:
                false_positive[n] += 1
                
        substitution[n] = min(false_negative[n], false_positive[n])
        deletion[n] = max(0, false_negative[n] - false_positive[n])
        insertion[n] = max(0, false_positive[n] - false_negative[n])
        
        n_active[n] = np.sum(true_col)
        
    sum_tp = np.sum(true_positive)
    sum_fp = np.sum(false_positive)
    sum_fn = np.sum(false_negative)
    
    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    error_rate = (np.sum(substitution) + np.sum(deletion) + np.sum(insertion)) / np.sum(n_active) if np.sum(n_active) > 0 else 0
    
    return f1_score, error_rate
