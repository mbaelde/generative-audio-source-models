import numpy as np
# For audio read/write, standard scipy handles wav. For mp3, pydub or librosa is often used.
# import librosa
import matplotlib.pyplot as plt

def analysis_simco():
    """
    Algorithm Analysis SimCO
    Translated from analysis_simco.m
    """
    p = 50
    m = 25
    l = 21
    n = 50000
    
    omega = np.random.randn(p, m)
    omega = (omega - np.mean(omega, axis=1, keepdims=True)) / np.std(omega, axis=1, keepdims=True)
    
    Y = np.random.randn(m, n)
    Y = (Y - np.mean(Y, axis=1, keepdims=True)) / np.std(Y, axis=1, keepdims=True)
    
    YYp = Y @ Y.T
    
    # Audio data application
    # In python: sound, fs = librosa.load('...', sr=None)
    # Y_audio = np.abs(librosa.stft(sound, n_fft=1024, hop_length=512, window='hann'))
    
    iter_max = 100
    t = 1e-3
    for iteration in range(iter_max):
        # Update X
        X = omega @ Y
        for nn in range(n):
            data = X[:, nn]
            idx = np.argsort(data)
            X[idx[:l], nn] = 0
            
        # Update omega
        H = 2 * X @ Y.T - 2 * omega @ YYp
        h_bar = np.zeros_like(H)
        omega_t = np.zeros_like(omega)
        
        for j in range(p):
            h_bar[j, :] = H[j, :] - H[j, :] * np.dot(omega[j, :], omega[j, :])
            norm_h = np.sum(h_bar[j, :] ** 2)
            
            if norm_h == 0:
                omega_t[j, :] = omega[j, :]
            else:
                omega_t[j, :] = omega[j, :] * np.cos(norm_h * t) + (h_bar[j, :] / norm_h) * np.sin(norm_h * t)
                
        omega = omega_t
        print(f"SimCO Iteration {iteration+1}/{iter_max}")

if __name__ == "__main__":
    analysis_simco()
