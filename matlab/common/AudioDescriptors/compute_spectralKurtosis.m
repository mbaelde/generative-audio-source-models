function SpectralKurtosis = compute_spectralKurtosis(freq_v, p_v, centroid, spread)

SpectralKurtosis = sum( (freq_v - centroid).^4 .* p_v) / spread^4;