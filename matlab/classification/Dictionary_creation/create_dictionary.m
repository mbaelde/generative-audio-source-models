%%
clear
clc
data_folder = '../../Data/A-Volute/';
addpath(genpath('AudioDescriptors'))
addpath(genpath('Statistics'))
addpath(genpath('Features'))
addpath(genpath('Results'))
%gcp
warning off

%%
dico = 2;
folder_feat = 'A-Volute/';
folder_interval = 'Interval/';

%% Define parameters for mixture models estimation of the spectrum
fs = 44100;                                         % Sampling rate
D_s = 0.01;                                         % Time interval between two analysys windows (sec)

T = 2.^(9:15);
D = round(2^nextpow2(D_s*fs));                      % Time interval between two analysys windows (sample)

N_fft = T;                                          % FFT size of the analysis window
N_spect = round(N_fft/5);                           % Number of points kept in the spectrum
criterion = 'BIC';                                  % Criterion used for choosing number of mixture components
verbose = 0;                                        % Flag to display information during estimation process
M_set = 1:20;                                       % Candidate values for the number of mixture components

param.T = T;
param.D = D;
param.N_spect = N_spect;
param.M_set = M_set;
param.criterion = criterion;
param.verbose = verbose;

%class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
my_names = dir(data_folder);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end

n_class = length(class);

%% Estimate dictionnaries
disp(['Computing dictionnary for T = ', num2str(T(dico))])
param.T = T(dico);
param.N_spect = N_spect(dico);
database = [];
cnt = 1;
for n = 1:n_class % class

disp(['Currently: class ', class{n}])
folder = [data_folder,class{n}];
names = dir(folder);
names = names(3:end);
n_sounds = length(names);

idx_sursample = [];
for i = 1:n_sounds
    file = [folder,'/',names(i).name];
    [sound,fs_sound] = audioread(file);
    % Test if the sound has the same sampling rate as the target
    f_factor = fs / fs_sound;
    if f_factor < 1
        idx_sursample = [idx_sursample, i];
        continue;
    end
    if f_factor ~= 1
        sound = resample(sound, f_factor, 1);
    end
    % Mean a stereo signal to form a mono signal
    if size(sound,2) == 2
        sound = mean(sound,2);
    end
    sound = sound - mean(sound);
    % Add noise if the signal is shorter than T, and add noise before the
    % beginning and after the end
    N_aux = length(sound);
    if N_aux <= T(dico) + D
        sound = [sound; 0.05*max(abs(sound))*randn(D+T(dico)-N_aux,1)];
    end
    sound = [0.05*max(abs(sound))*randn(D,1); sound; 0.05*max(abs(sound))*randn(D,1)];

    N = floor((length(sound) - T(dico)) / D);
    for b = 1:N
        database(cnt,:) = [sound((b-1)*D+1:(b-1)*D+T(dico))',n,b];
        cnt = cnt + 1;
    end
    
%     signal{cnt}.signal = sound;
%     signal{cnt}.class = n;
%     N_s = size(signal{cnt}.signal,1);

    %[feature{cnt},error] = compute_signal_feature(signal{cnt}, fs, param, folder_interval(1:end-1));
    %if ~isempty(error)
%         fileID = fopen(['log/signal_',names(i).name(1:end-4),'.txt'],'w');
%         fprintf(fileID,error);
%         fclose(fileID);
       %save(['log/signal_',num2str(i),'.txt'],'error')
   % end
    
    progressbar(i,n_sounds)
end
end
for k = 1:n_class
    idx{k} = find(database(:,end-1) == k);
end
aux_database = database;
aux_database(union(union(idx{1},idx{5}),idx{7}),end-1) = 1; % Engine
aux_database(union(idx{3},idx{4}),end-1) = 2; % Detonation
aux_database(union(idx{8},idx{9}),end-1) = 3; % Voice
aux_database(idx{2},end-1) = 4; % Alarm
aux_database(idx{6},end-1) = 5; % Step 
database = [];
for k = 1:5
    database = [database; aux_database(aux_database(:,end-1) == k,:)];
end
clear aux_database
save(['Database/A-Volute/FS',num2str(fs),'/T',num2str(T(dico)),'/Metaclasse/database_overlap.mat'],'database')
%
%save(['Features/',folder_feat,folder_interval,'T',num2str(T(dico)),'/feature_',class{n},'.mat'],'feature')
%end
%%
aux_feature = [];
for k = 1:n_class
    clear feature
    load(['Features/',folder_feat,folder_interval,'T',num2str(T(dico)),'/feature_',class{k},'.mat'])
    aux_feature = [aux_feature, feature];
end
feature = aux_feature;
save(['Features/',folder_feat,folder_interval,'T',num2str(T(dico)),'/feature.mat'],'feature')
