function y = cauchyrnd(n,a)
%% cauchyrnd
%
% This function samples n points of the Cauchy distribution of parameter a.
%
% Author: Maxime Baelde
% A-Volute // 2016
u = rand(n,1);
y = a*tan(pi*(u-0.5));
