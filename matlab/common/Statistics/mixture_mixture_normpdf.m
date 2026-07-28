function out = mixture_mixture_normpdf(x, mu, sigma, weight, prop_mixture)
%% mixture_mixture_normpdf
%
% This function returns a mixture of mixture normal pdf of mu and sigma,
% with proportions weight within each mixture and prop_mixture for the
% mixture of mixture, over points x.
%
% Author: Maxime Baelde
% A-Volute // 2016
K = length(prop_mixture);

out = 0;
for k = 1:K
    out = out + prop_mixture(k)*mixture_normpdf(x,mu(k,:), sigma(k,:), weight(k,:));
end