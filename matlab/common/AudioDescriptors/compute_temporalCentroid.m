function TemporalCentroid = compute_temporalCentroid(env_energy_data, t)
%% compute_temporalCentroid
%
% This function returns the temporal centroid of an audio data.
%
% TemporalCentroid = compute_temporalCentroid(env_energy_data, t)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

% The temporal centroid is the center of gravity of the energy envelope
TemporalCentroid = sum( env_energy_data .* t ) / sum(env_energy_data);