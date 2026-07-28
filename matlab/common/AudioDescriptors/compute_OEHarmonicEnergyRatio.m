function OEHarmonicEnergyRatio = compute_OEHarmonicEnergyRatio(harmonic_model)

ampl = harmonic_model.ampl;
H = length(ampl);

OEHarmonicEnergyRatio = ( sum(ampl(1:2:H).^2) ) / ( sum(ampl(2:2:H).^2) );