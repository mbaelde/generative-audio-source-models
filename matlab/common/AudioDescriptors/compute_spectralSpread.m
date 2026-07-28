function SpectralSpread = compute_spectralSpread(freq_v, p_v, centroid)

SpectralSpread = sqrt(sum( (freq_v - centroid).^2 .* p_v));