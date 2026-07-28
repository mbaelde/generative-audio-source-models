function SpectralSkewness = compute_spectralSkewness(freq_v, p_v, centroid, spread)

SpectralSkewness = sum( (freq_v - centroid).^3 .* p_v) / spread^3;