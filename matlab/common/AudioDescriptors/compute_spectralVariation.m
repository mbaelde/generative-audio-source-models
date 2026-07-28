function SpectralVariation = compute_spectralVariation(stft_data)
%% compute_spectralVariation
%
% This function returns the spectral flux of an audio data.
%
% SpectralVariation = compute_spectralVariation(stft_data)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

% The spectral variation is the amount of variation of the spectrum over
% time.
SpectralVariation = 1 - sum(stft_data(:,1:end-1) .* stft_data(:,2:end)) ./ (sqrt(sum(stft_data(:,1:end-1).^2)) .* sqrt(sum(stft_data(:,2:end).^2)));
