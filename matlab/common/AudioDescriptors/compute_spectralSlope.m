function SpectralSlope = compute_spectralSlope(amp_stft, freq_v)

K = length(freq_v);

SpectralSlope = ( 1 / (sum(amp_stft))) * ((K * sum(freq_v .* amp_stft) - sum(freq_v) * sum(amp_stft)) / (K * sum(freq_v.^2) - sum(freq_v)^2));
