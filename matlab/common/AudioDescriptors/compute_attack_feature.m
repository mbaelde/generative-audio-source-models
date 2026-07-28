function [attack_time, attack_slope, decrease_slope, effective_duration, energy_modulation] = compute_attack_feature(env_energy_data, fs)
%% compute_attack_feature
%
% This function returns different features related to the attack of an
% audio data. It uses the energy envelope of the data sampled at fs.
%
% [attack_time, attack_slope, decrease_slope, effective_duration, energy_modulation] = compute_attack_feature(env_energy_data, fs)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

% Vector of time instants
time = 0:1/fs:(length(env_energy_data)-1)/fs;

% Coefficients of the weakest-effort method 
alpha = 3;      % alpha
th = 0.1:0.1:1; % thresholds

% Maximum energy 
max_energy = max(env_energy_data);

% Time indeces where the envelope goes above each threshold of the maximum energy
idx = zeros(1,10);
for i = 1:10
    idx(i) = find(env_energy_data >= th(i) * max_energy,1);
end

% Convert the indeces in time instants
t = time(idx);
% Compute the efforts
w = diff(t);
%% Attack Time
% Compute the average efforts
w_ = mean(w);

% Detect the starting time of the attack: find the first effort that goes
% below alpha * w_
th_st = find(w < alpha * w_,1);
% Same for the end time
th_end = 10-find(fliplr(w) < alpha * w_,1);

% Store them in the attack_time
attack_time = t([th_st,th_end]);

%% Attack Slope
% The attacki slope is the average temporal slope of the energy during the
% attack segment.

% Compute the local slopes
slope = diff(env_energy_data(idx))' ./ w;
% Compute weights
weight = 0.5 * normpdf(th(1:end-1)-0.5);
% Weigthed average of the local slopes
attack_slope = sum( slope .* weight ) / sum(weight);

%% Decrease Slope
idx = find(env_energy_data == max_energy);
t_max = time(idx);

covET = cov(log(env_energy_data(idx:end)), time(idx:end));
decrease_slope = covET(1,2) / var(time(idx:end));

%% Effective duration
% It is the time the energy envelope is above 40% the maximum of energy
idx(1) = find(env_energy_data > 0.4 * max_energy,1);
idx(2) = length(env_energy_data) - find(flipud(env_energy_data) > 0.4 * max_energy,1);

effective_duration = diff(time(idx));

%% Energy modulation
% It is the sinusoidal part of the sustain part of the energy envelope.
idx = find(env_energy_data == max_energy);
% Model the sustain part
energy_model = max_energy * exp(-decrease_slope * (time(idx:end) - t_max))';
% Compute the residual
residual = env_energy_data(idx:end) - energy_model;
% Take the maximum of the amplitude spectrum
freq = 0:fs/length(residual):(length(residual)-1)*fs/length(residual);
fft_residual = abs(fft(residual));
fft_residual = fft_residual(1:find(freq > 10,1));

if ~isempty(max(fft_residual))
    energy_modulation.amp = max(fft_residual);
    energy_modulation.f = freq(fft_residual == max(fft_residual));
else
    energy_modulation.amp = 0;
    energy_modulation.f = 0;
end
