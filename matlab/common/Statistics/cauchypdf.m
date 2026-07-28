function y = cauchypdf(x,a)
%% cauchypdf
%
% This function returns the Cauchy pdf of parameters a over points x.
%
% Author: Maxime Baelde
% A-Volute // 2016
y = a./(pi*(a*a+x.*x));
