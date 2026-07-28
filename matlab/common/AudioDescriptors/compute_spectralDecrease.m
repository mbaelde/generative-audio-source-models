function SpectralDecrease = compute_spectralDecrease(amp_stft)

K = length(amp_stft);

SpectralDecrease = sum((amp_stft(2:end) - amp_stft(1)) ./ (1:K-1)') / sum(amp_stft(2:end));