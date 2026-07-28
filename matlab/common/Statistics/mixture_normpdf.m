function out = mixture_normpdf(x, mu, sigma, weight)
%% mixture_normpdf
%
% This function returns a mixture of normal pdf of mu and sigma,
% with proportions weight, over points x.
%
% Author: Maxime Baelde
% A-Volute // 2016

M = length(weight);

out = zeros(size(x));
for m = 1:M
    out = out + weight(m) .* normpdf(x, mu(m), sigma(m));
end