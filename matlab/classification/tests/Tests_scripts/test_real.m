clear
clc
data_folder = '../../Data/';
folder_file = 'Test/';
addpath(genpath('AudioDescriptors'))
addpath(genpath('Statistics'))
addpath(genpath('Features'))
addpath(genpath('Results'))
warning off

%% Initialisation
fs = 44100;                    % Sampling rate
D_s = 0.01;                    % Time interval between two analysys windows (sec)

T = [512, 1024, 2048, 4096];
D = round(2^nextpow2(D_s*fs)); % Time interval between two analysys windows (sample)

N_fft = T;                     % FFT size of the analysis window
N_spect = round(N_fft/5);      % Number of points kept in the spectrum

class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
n_class = length(class);

folder_feat = 'Clean database features/';
folder_interval = 'Interval/';
type = '';

%% Initialise
my_class = 1:length(class);

dico = 3;
% Load preprocess dictionnary
load(['Features/',folder_feat,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_T',num2str(T(dico)),'.mat']);

winsize = T(dico);

f = 0:fs/N_fft(dico):fs/2;
f = f(1:N_spect(dico));

param.N_spect = N_spect(dico);
param.winsize = winsize;
param.fs = fs;
param.type = 'clavel';
param.f = f;
param.snr = inf;
param.gpuFlag = 1;
param.th_silent = 1e-3;

%%
[sound,fs] = audioread([data_folder,folder_file,'Maelstrom Demo [01 Front Left].wav']);

test_class = zeros(length(sound),1);
idx_test = [0    , 5    , 0;
            5    , 20   , 9;
            20   , 24   , 0;
            24   , 51   , 9;
            51   , 73.5 , 9;
            73.5 , 74.5 , 4;
            74.5 , 88   , 9;
            88   , 100  , 0;
            100  , 115  , 9;
            115  , 132  , 0;
            132  , 147  , 9;
            147  , 226  , 0;
            148.5, 150  , 3;
            150  , 154  , 4;
            154  , 163  , 7;
            156  , 157  , 4;
            157  , 158  , 9;
            158  , 163  , 4;
            167  , 170  , 4;
            169  , 170  , 9;
            172  , 189  , 7;
            190  , 192  , 3;
            193.5, 226  , 9];
        
for ii = 1:size(idx_test,1)
    t = (idx_test(ii,1)*fs:idx_test(ii,2)*fs)+1;
    test_class(t) = idx_test(ii,3);
end
% Mean a stereo signal to form a mono signal
if size(sound,2) == 2
    sound = mean(sound,2);
end
sound = sound - mean(sound);
winsize = T(dico);
hopsize = D;
N_test = length(sound);
N_buffer = floor((N_test-winsize)/hopsize);

database = zeros(N_buffer,winsize+2);
for b = 1:N_buffer
    database(b,1:end-2) = sound(1+(b-1)*hopsize:(b-1)*hopsize+winsize);
    database(b,end-1) = test_class(b*hopsize);
end

%%
prior_g = zeros(1,n_class);
for k = 1:n_class
    prior_g(k) = sum(aux_L(:,end-1) ==k);
end

tic
L_bay = identification_general(database, aux_L, prior_g, my_class, param);
elapsed_time{k} = toc / prior_g(k);
%%  
    idx_p = aux_L_bay > 0;
    aux_L_bay = aux_L_bay(idx_p);
    true_class = [];
    for k = 1:n_class
        true_class = [true_class, k*ones(1,sum(idx_test{k}))];
    end
    true_class = true_class(idx_p);
    confusion_matrix = confusionmat(true_class, aux_L_bay, 'order', 1:n_class);
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class]);
    save(['Results/Performance/Comp Clavel/Folds 80 train 20 test/ESC-50/resul_dataset_Models_3_fold_',num2str(num_base),'.mat'],'L_bay', 'aux_L_bay', 'BAY_confusionmatrix')

