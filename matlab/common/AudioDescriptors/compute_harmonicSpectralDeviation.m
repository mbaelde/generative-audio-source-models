function HarmonicSpectralDeviation = compute_harmonicSpectralDeviation(harmonic_model)

ampl = harmonic_model.ampl;
H = length(ampl);

SE = ampl;

for h = 2:H-1
    SE(h) = mean(ampl(h-1:h+1));
end

HarmonicSpectralDeviation = mean(ampl - SE);