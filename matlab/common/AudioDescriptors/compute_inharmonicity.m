function Inharmonicity = compute_inharmonicity(harmonic_model)

f = harmonic_model.f;
ampl = harmonic_model.ampl;
H = length(f);

Inharmonicity = (2 * f(1)) * ( ( sum((f - (1:H)*f(1)).* (ampl.^2)) ) / ( sum(ampl.^2) ) );