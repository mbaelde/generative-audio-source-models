function mixture_cdf = model2cdf(mixture_model, f, normalize)
%% model2pdf
%
% This functions returns a mixture model pdf based on the mixture model
% parameters in mixture_model and points in f, and normalize it w.r.t. the
% number of frequency bins.
%
% Author: Maxime Baelde
% A-Volute // 2016
N_spect = length(f)-1;
mixture_cdf = mixture_normcdf(f, mixture_model.mu, sqrt(mixture_model.sigma), mixture_model.mixing_coeff);
if normalize
    mixture_cdf = N_spect * mixture_cdf / sum(mixture_cdf);
end