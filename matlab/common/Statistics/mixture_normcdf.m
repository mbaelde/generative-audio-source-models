function out = mixture_normcdf(x, mu, sigma, weight)
%% mixture_normcdf
%
% This function returns a mixture of normal cdf of mu and sigma,
% with proportions weight, over points x.
%
% Author: Maxime Baelde
% A-Volute // 2016
M = length(weight);

out = 0;
for m = 1:M
    out = out + weight(m) .* normcdf(x, mu(m), sigma(m));
end