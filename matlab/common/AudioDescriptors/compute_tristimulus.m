function tristimulus = compute_tristimulus(harmonic_model)


ampl = harmonic_model.ampl;

tristimulus = [ampl(1), sum(ampl(2:4)), sum(ampl(5:end))] / sum(ampl);