function ACcoefs = compute_AutoCorrelation(buffer,winsize)
%% compute_AutoCorrelation
%
% This function returns the first 12 normalized autocorrelation 
% coefficients of an audio data.
%
% ACcoefs = compute_AutoCorrelation(buffer,winsize)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

c = xcorr(buffer);
ACcoefs = c(winsize+1:winsize+12) / c(winsize);