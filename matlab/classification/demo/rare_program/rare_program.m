function [labels, durations, gcr, confusion_matrix] = rare_program(sound_name, true_label)
load('tree_69732.mat')
% Global parameters
T = 2048;                       % Window size
N_spect = 410;                  % Number of bins
n_buff = 10;                    % Number of buffers to aggregate for final decision

% Read sound
[sound,fs] = audioread(sound_name);
% Resample if not sampled at 44100Hz
if fs ~= 44100
    f_factor = fs / 44100;
    if f_factor < 1
        sound = resample(sound, 1/f_factor, 1);
    elseif f_factor > 1
        sound = resample(sound, 1, f_factor);
    else
        fprintf('Error: sampling rate must be a multiple of 44100Hz')
    end
end

% Convert to mono, suppose sound is a matrix (n_channel, length)
if size(sound,1) > size(sound,2)
    sound = sound';
end
sound = mean(sound,1);
% Length of the sound
N = length(sound);
% Add silence if the sound is not long enough
if N < T * n_buff
    sound = [sound, zeros(1,T * n_buff - N)];
end
N = length(sound);

% Split the sound into buffers
N_buffer = floor(N / T);
database = zeros(N_buffer, T);
for n = 1:N_buffer
    database(n,:) = sound( (n-1)*T+1:n*T );
end

% Compute the decisions
param.N_spect = N_spect;
param.n_buff = n_buff;
[labels,durations] = identification_tree_incomplete(database, my_tree, subtree, param);

% if the label is known
if ~isnan(true_label)
    if length(true_label) == 1
        true_class = true_label * ones(1,N_buffer);
    end
    % Take only the portion of signal where the decision is computed
    idx_p = labels > 0;
    labels = labels(idx_p);
    true_class = true_class(idx_p);
    durations = durations(idx_p);
    % Number of classes predicted by the algorithm
    n_class_predicted = length(unique(labels));
    % Compute metrics
    confusion_matrix = confusionmat(true_class, labels);
    for k = 1:n_class_predicted
        if sum(confusion_matrix(k,:)) ~= 0
            confusion_matrix(k,:) = confusion_matrix(k,:) ./ sum(confusion_matrix(k,:),2);
        end
    end
    % Good recognition rate per class
    gcr = diag(confusion_matrix)*100;
else
    gcr = NaN;
    confusion_matrix = NaN;
end