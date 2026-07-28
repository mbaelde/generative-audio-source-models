function y = bilaplace(x,mu1,a1,mu2,a2,p)
%% bilaplace
%
% This function returns bilaplace pdf of parameters mu1, a1 and mu2, a2,
% with proportion p, over points x.
%
% Author: Maxime Baelde
% A-Volute // 2016
y = p*(1/2*a1)*exp(-abs(x-mu1)/a1)+(1-p)*(1/2*a2)*exp(-abs(x-mu2)/a2);