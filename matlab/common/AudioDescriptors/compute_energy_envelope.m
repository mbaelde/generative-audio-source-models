function energy_envelope = compute_energy_envelope(data, fs, fc)
%% compute_energy_envelop
%
% This function returns the energy envelope of an audio data, sampled at a
% sampling rate fs. It filter the Hilbert Transform of data at a cutoff 
% frequency fc.
%
% energy_envelop = compute_energy_envelop(data, fs, fc)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

% Take the Hilbert transform of data
analytical_data = hilbert(data);

% Get the amplitude of the analytical signal
amp_analytical = abs(analytical_data);

% Design a Butterworth filter of order 3, with cutoff frequency fc
[b,a] = butter(3,2*fc/fs);

% Filter the analytical signal to obtain the energy envelop
energy_envelope = filter(b,a,amp_analytical);
