function SpectralCrest = compute_spectralCrest(amp_stft)

SpectralCrest = max(amp_stft) / mean(amp_stft);