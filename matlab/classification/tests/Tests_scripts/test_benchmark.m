clear
clc
warning off
%%
addpath(genpath('AudioDescriptors'))
addpath(genpath('ToolboxBenchmark'))
addpath(genpath('Results benchmark'))
addpath(genpath('Database'))
data_folder = '../../Data/ESC-50/';

%class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
my_names = dir(data_folder);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end
n_class = length(class);

fs = 44100;
T = [512,1024,2048,4096];

% %% Create database
% cnt = 1;
% for k = 1:length(class)
%     folder = [data_folder,'/',class{k}];
%     names = dir(folder);
%     names = names(3:end);
%     for n = 1:length(names)
%         [sound, fs_sound] = audioread([folder,'/',names(n).name]);
%         f_factor = fs / fs_sound;
%         if f_factor ~= 1
%             sound = resample(sound, f_factor, 1);
%         end
%         if size(sound,2) == 2
%             sound = mean(sound,2);
%         end
%         database{cnt}.sound = sound;
%         database{cnt}.fs = fs;
%         database{cnt}.class = k;
%         cnt = cnt + 1;
%     end
%     progressbar(k,length(class))
% end
% save('database.mat', 'database')
%% C. Clavel, Event detection for an audio-based surveillance system, 2005
dico = 3;
%for num_base = 1:5;
num_base = 5;
type_data = 'Models';
if num_base == 0
    if dico == 1
        load(['Database/T',num2str(T(dico)),'/database.mat'])
    else
        load(['Database/T',num2str(T(dico)),'/database_overlap.mat'])
    end
else
    %load(['Database/T',num2str(T(dico)),'/On ',type_data,'/dataset_T',num2str(T(dico)),'_fold_',num2str(num_base),'.mat'])
    load(['Database/ESC-50/T',num2str(T(dico)),'/On ',type_data,'/dataset_T',num2str(T(dico)),'_fold_',num2str(num_base),'.mat'])
end
clear aux_L_training aux_L_test
clear descriptors infos input

database_training = [];
database_test = [];

for k = 1:n_class
    disp(['Currently: class ', class{k}])
    load(['Database\ESC-50\T',num2str(T(dico)),'\database_overlap_',class{k},'.mat'])
    database_test = [database_test; database(idx_test{k},:)];
    database_training = [database_training; database(idx_train{k},:)];
end
save(['database_training_fold_',num2str(num_base),'.mat'],'database_training')
save(['database_test_fold_',num2str(num_base),'.mat'],'database_test')

if num_base == 0
    input.database.training = database;
    input.database.test = [];
else
    input.database.training = database_training;
    input.database.test = database_test;
end
infos.fs = fs;
clear database database_training database_test

descriptors.descriptor{1} = {'Energy', 1};
descriptors.descriptor{2} = {'MFCC', 8};
descriptors.descriptor{3} = {'SpectralCentroid', 1};
descriptors.descriptor{4} = {'SpectralSpread', 1};
descriptors.pca = [1, 13];
descriptors.delta = 2;
input.descriptors = descriptors;

infos.winsize = T(dico)/fs; % s
infos.overlap = 0.5; % percentage
input.infos = infos;

input.method = 'GMM-Clavel';

result_clavel = benchmark_classification(input);
if num_base == 0
    save(['Results benchmark/result_clavel_dataset_',num2str(dico),'.mat'], 'result_clavel')
else
    save(['Results benchmark/On ',type_data,'/ESC-50/result_clavel_dataset_',num2str(dico),'_fold_',num2str(num_base),'.mat'], 'result_clavel')
end
%end
%% R. Radhakrishnan, Audio Analysis for Surveillance Applications, 2005
dico = 4;
load(['Database/T',num2str(T(dico)),'/database.mat'])
clear descriptors infos input

input.database.training = database;
infos.fs = fs;

% test_file = load('test_file.mat');
% test_file = test_file.test_file;
% test_sound = test_file.test_sound;
% test_class = test_file.test_class;
%
% input.database.test{1}.sound = test_sound;
% input.database.test{1}.fs = fs;
% input.database.test{1}.class = test_class;

input.database.test = [];

descriptors.descriptor{1} = {'MFCC', 12};
descriptors.pca = [0, 0];
descriptors.delta = 0;
input.descriptors = descriptors;

infos.winsize = T(dico)/fs; % s
infos.overlap = 0.5; % percentage
input.infos = infos;

input.method = 'GMM';

result_radhakrishnan = benchmark_classification(input);
save(['Results benchmark/result_radhakrishnan_dico_',num2str(dico),'.mat'], 'result_radhakrishnan')
%% D. Istrate, Real-Time Sound Analysis for Medical Remote Monitoring, 2008
dico = 4;
load(['Database/T',num2str(T(dico)),'/database.mat'])
clear descriptors infos input

input.database.training = database;
infos.fs = fs;

% test_file = load('test_file.mat');
% test_file = test_file.test_file;
% test_sound = test_file.test_sound;
% test_class = test_file.test_class;
%
% input.database.test{1}.sound = test_sound;
% input.database.test{1}.fs = fs;
% input.database.test{1}.class = test_class;

input.database.test = [];

descriptors.descriptor{1} = {'MFCC', 16};
descriptors.descriptor{2} = {'SpectralCentroid', 1};
descriptors.descriptor{3} = {'ZCR', 1};
descriptors.descriptor{4} = {'SpectralRolloff', 1};
descriptors.pca = [0, 0];
descriptors.delta = 0;
input.descriptors = descriptors;

infos.winsize = T(dico)/fs; % s
infos.overlap = 0.5; % percentage
input.infos = infos;

input.method = 'GMM';

result_istrate = benchmark_classification(input);
save('Results benchmark/result_istrate.mat', 'result_istrate')

%% General method : GMM
clear descriptors infos input

input.database.training = database;

input.database.test = [];

% descriptors.descriptor{1} = {'Attack',1};
descriptors.descriptor{1} = {'Autocorrelation',12};         % //
descriptors.descriptor{2} = {'ZCR',1};                      % //
descriptors.descriptor{3} = {'Energy', 1};                  % //
descriptors.descriptor{4} = {'SpectralCentroid', 1};        % //
descriptors.descriptor{5} = {'SpectralSpread', 1};          % //
descriptors.descriptor{6} = {'SpectralSkewness',1};         % //
descriptors.descriptor{7} = {'SpectralKurtosis',1};         % //
descriptors.descriptor{8} = {'SpectralSlope',1};            % //
descriptors.descriptor{9} = {'SpectralDecrease',1};         % //
descriptors.descriptor{10} = {'SpectralRolloff',1};         % //
descriptors.descriptor{11} = {'SpectralFlatness',1};        % //
descriptors.descriptor{12} = {'SpectralCrest',1};           % //
descriptors.descriptor{13} = {'MFCC', 24};                   % //
descriptors.descriptor{14} = {'f0',1};                      % //
descriptors.descriptor{15} = {'HarmonicEnergy',1};          % //
descriptors.descriptor{16} = {'NoiseEnergy',1'};            % //
descriptors.descriptor{17} = {'Noisiness',1};               % //
descriptors.descriptor{18} = {'Tristimulus',3};             % //
descriptors.descriptor{19} = {'Inharmonicity',1'};          % //
descriptors.descriptor{20} = {'HarmonicSpectralDeviation',1};%//
descriptors.descriptor{21} = {'OEHarmonicEnergyRatio',1};   %//
% descriptors.descriptor{22} = {'TemporalCentroid',1};

descriptors.pca = [0, 00];
descriptors.delta = 0;
input.descriptors = descriptors;

infos.winsize = 0.020; % s
infos.overlap = 0.5; % percentage
input.infos = infos;

input.method = 'GMM';


result_gmm = benchmark_classification(input);
save(['Results benchmark/result_gmm_diff',num2str(descriptors.delta),'.mat'], 'result_gmm')
%save(['Results benchmark/result_gmm_pca',num2str(descriptors.pca(2)),'_diff',num2str(descriptors.delta),'.mat'], 'result_gmm')

%% General method : SVM
dico = 1;
load(['Database/T',num2str(T(dico)),'/database.mat'])
clear descriptors infos input

input.database.training = database;
infos.fs = fs;

% test_file = load('test_file.mat');
% test_file = test_file.test_file;
% test_sound = test_file.test_sound;
% test_class = test_file.test_class;
%
% input.database.test{1}.sound = test_sound;
% input.database.test{1}.fs = fs;
% input.database.test{1}.class = test_class;

input.database.test = [];

descriptors.descriptor{1} = {'Energy', 1};
descriptors.descriptor{2} = {'MFCC', 8};
descriptors.descriptor{3} = {'SpectralCentroid', 1};
descriptors.descriptor{4} = {'SpectralSpread', 1};
descriptors.pca = [1, 20];
descriptors.delta = 2;
input.descriptors = descriptors;

infos.winsize = T(dico)/fs; % s
infos.overlap = 0.5; % percentage
input.infos = infos;

input.method = 'SVM';

result_svm = benchmark_classification(input);
save(['Results benchmark/result_svm_dico_',num2str(dico),'.mat'], 'result_svm')


%% Neural Network
dico = 1;
load(['Database/T',num2str(T(dico)),'/database.mat'])
clear descriptors infos input

input.database.training = database;
infos.fs = fs;

% test_file = load('test_file.mat');
% test_file = test_file.test_file;
% test_sound = test_file.test_sound;
% test_class = test_file.test_class;
%
% input.database.test{1}.sound = test_sound;
% input.database.test{1}.fs = fs;
% input.database.test{1}.class = test_class;

input.database.test = [];

descriptors.descriptor{1} = {'Energy', 1};
descriptors.descriptor{2} = {'MFCC', 8};
descriptors.descriptor{3} = {'SpectralCentroid', 1};
descriptors.descriptor{4} = {'SpectralSpread', 1};
descriptors.pca = [1, 20];
descriptors.delta = 2;
input.descriptors = descriptors;

infos.winsize = T(dico)/fs; % s
infos.overlap = 0.5; % percentage
input.infos = infos;

input.method = 'NN';

result_clavel = benchmark_classification(input);
save(['Results benchmark/result_clavel_dico_',num2str(dico),'.mat'], 'result_clavel')

%% Create test file
test_sound = [];
test_class = [];
for k = [1,2,3,4,5,6,7,8,9]
    folder = [data_folder,'/',class{k}];
    names = dir(folder);
    names = names(3:end);
    idx = randperm(length(names),1);
    %[aux,fs_sound] = audioread([folder,'/',names(idx).name]);
    [aux,fs_sound] = audioread([folder,'/',names(1).name]);
    f_factor = fs / fs_sound;
    if f_factor ~= 1
        aux = resample(aux, f_factor, 1);
    end
    if size(aux,2) == 2
        aux = mean(aux,2);
    end
    aux = aux - mean(aux);
    silence = randi([2048,4096]);
    test_sound = [test_sound; zeros(1024+silence,1); aux];
    test_class = [test_class; zeros(1024+silence,1); k*ones(size(aux))];
    test_idx(k) = idx;
end
test_file.test_sound = test_sound;
test_file.test_class = test_class;
test_file.test_idx = test_idx;
save('test_file_known_3.mat', 'test_file');
%%
test_sound = [];
test_class = [];
test_idx = [];
idx_1 = 1;
idx_2 = 1;
for k = 1:length(class)
    folder = [data_folder,'/',class{k}];
    names = dir(folder);
    names = names(3:end);
    %idx = 1:length(names);
    %     if k == 4
    %         idx = idx_1;
    %         idx_1 = idx_1 + 1;
    %     end
    %     if k == 6
    %         idx = idx_2;
    %         idx_2 = idx_2 + 1;
    %     end
    %for i = idx
    i = randi([1,length(names)]);
    [aux,fs_sound] = audioread([folder,'/',names(i).name]);
    f_factor = fs / fs_sound;
    if f_factor ~= 1
        aux = resample(aux, f_factor, 1);
    end
    if size(aux,2) == 2
        aux = mean(aux,2);
    end
    aux = aux - mean(aux);
    silence = randi([2048,4096]);
    test_sound = [test_sound; zeros(1024+silence,1); aux];
    test_class = [test_class; zeros(1024+silence,1); k*ones(size(aux))];
    test_idx = [test_idx;i];
    % end
end
test_file.test_sound = test_sound;
test_file.test_class = test_class;
test_file.test_idx = test_idx;
save('Test files/test_file_known_6.mat', 'test_file');