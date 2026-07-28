function TotalEnergy = compute_totalEnergy(amp_stft)

TotalEnergy = sum(amp_stft.^2);