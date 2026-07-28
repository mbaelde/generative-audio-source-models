function SpectralCentroid = compute_spectralCentroid(freq_v, p_v)

SpectralCentroid = sum(freq_v .* p_v);