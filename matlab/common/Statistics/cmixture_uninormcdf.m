function out = cmixture_uninormcdf(x, mu, sigma, weight)
%% cmixture_uninormpdf
%
% This function returns c.d.f. of a mixture of normals distributions. It
% evaluates the c.d.f. at points in x, with components parameters weight
% (proportions), mu (means) and sigma (standard deviations).
%
% out = cmixture_uninormcdf(x, mu, sigma, weight, M)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA
out = zeros(size(x));
M = length(mu);

for m = 1:M
    out = out + weight(m) .* normcdf(x, mu(m), sigma(m));
end