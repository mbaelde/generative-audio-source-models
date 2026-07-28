function SpectralFlatness = compute_spectralFlatness(amp_stft)

SpectralFlatness = geomean(amp_stft) / mean(amp_stft);