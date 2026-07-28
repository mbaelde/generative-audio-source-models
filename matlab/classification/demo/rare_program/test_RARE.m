%% Test program for the RARE toolbox
% Author: Maxime BAELDE
% Date: 23/11/2016
% Company: A-Volute / INRIA
%% Mono
% Folder of stored sounds
data_folder = 'test_sounds';
% Add the folder to path (Matlab related)
addpath(data_folder)
% Get the names of the files and the corresponding classes
my_names = dir(data_folder);
my_names = my_names(3:end);
for n = 1:length(my_names)
    classes{n} = my_names(n).name(1:end-4);
end
% Class coding:
% 1: airplane
% 2: alarm
% 3: explosion
% 4: gunshot
% 5: helicopter
% 6: step
% 7: vehicule
% 8: voice (female)
% 9: voice (male)

for k = 1:length(classes)
    sound_name = my_names(k).name;
    true_label = k;
    [labels, durations, gcr, confusion_matrix] = rare_program(sound_name, true_label);
    figure(1)
    clf
    plot(true_label * ones(1,length(labels)), '-r')
    hold on
    plot(labels, '--b')
    legend('True class', 'Predicted class')
    disp('Press any key to continue...')
    pause
end

%% Poly
clear
%close all

% Folder of stored sounds
data_folder = 'test_sounds\';
% Add the folder to path (Matlab related)
addpath(data_folder)
addpath(genpath('jsonlab'))
% Get the names of the files and the corresponding classes

% Class coding:
% 1: detonation
% 2: step
% 3: voice

sound_name = 'battlefield_2';
[sound,fs] = audioread([sound_name,'.wav']);

param.T = 2048;
param.D = 1024;
param.fs = 44100;
param.n_fft = round(param.T / 5);
param.q = param.T / 2 + 1;
param.power = 2;
param.gpuFlag = 0;
param.threshold = -160;
param.dict = 0;
param.create_feature = 0;

[database,activation_matrix, class, gm] = create_dataset_poly(sound_name, data_folder, param);

param.gm = gm;

if param.create_feature
%     load('\\AUDIO06\ServeurMatlab\Ph. D Thesis\Softwares\Database\BF\Complete\audio_bf_no\feature_fs44100_N1025.mat','feature')
%     load('\\AUDIO06\ServeurMatlab\Ph. D Thesis\Softwares\Database\BF\Folds\Set 1\idx_train_test_fs44100_T2048.mat')
%     n_idx = 1;
%     fold = 1;
%     id_class = unique(feature(:,end-1));
%     feature_training = [];
%     for k = 1:n_class
%         feature_class = feature(feature(:,end-1) == id_class(k),:);
%         feature_training = [feature_training; feature_class(idx_train{fold}{k},:)];
%     end
%     clear feature_class
%     load('\\AUDIO06\ServeurMatlab\Ph. D Thesis\Softwares\Database\BF\Folds\Set 1\Z_fold_1_fs44100_T2048.mat')
%     th.value = 80;
%     th.type = 'f1';
%     th.initial_guess = [175, 46, 13, 114, 30, 18, 40];
%     N_sounds = optimize_reduce_size_poly(feature_training, Z, database, 10, th, param);
else
    load('dict_poly.mat')
end

id_class = setdiff(unique(database(:,end-1)),0);
if param.use_reduced == 1
	n_class = length(unique(feature_reduced(:,end-1)));
else
    n_class = length(unique(feature_training(:,end-1)));
end

prior_g = zeros(1,n_class);
for k = 1:n_class
    try
        prior_g(id_class(k)) = sum(database(:,end-1) == id_class(k));
    catch
    end
end
%%
feature_training = [];
for k = 1:n_class
feature_training = [feature_training;feature(feature(:,end-1) == k ,:)];
end

database_test = [];
for k = 1:n_class
database_test = [database_test;database(database(:,end-1) == k ,:)];
end

param.gpuFlag=1;
[posterior_g,computation_time] = identification(database, feature_reduced, prior_g, param);

m = 10;
N = size(posterior_g,1);
labels = zeros(1,N);
for b = 1:N
    if mod(b, m) == 0
        sum_g = sum(gather(posterior_g((b-m+1):b,:)),1,'omitnan');
        labels((b-m+1):b) = find(max(sum_g) == sum_g);
    end
end

true_class = database(:,end-1)';

% idx_zero = true_class == 0;
% true_class = true_class(~idx_zero);
% labels = labels(~idx_zero);
% idx_zero = labels == 0;
% true_class = true_class(~idx_zero);
% labels = labels(~idx_zero);

N = length(labels);

matrix_true = zeros(3, N);
matrix_pred = zeros(3, N);
for n = 1:N
    if true_class(n) == 0
        matrix_true(:,n) = 0;
    else
        matrix_true(unique(gm(true_class(n),:)),n) = 1;
    end
    if labels(n) == 0
        matrix_pred(:,n) = 0;
    else
        matrix_pred(unique(gm(labels(n),:)),n) = 1;
    end
end

figure(1)
clf
imagesc(matrix_true)
figure(2)
clf
imagesc(matrix_pred)

% Segment-based metrics
[f1_score, error_rate] = metrics_sed(matrix_true, matrix_pred)
mean(computation_time)

results.matrix_true = matrix_true;
results.matrix_pred = matrix_pred;
results.f1_score = f1_score;
results.error_rate = error_rate;
results.computation_time = mean(computation_time);
%%
json = savejson('results',results);
fileID = fopen(['results_',sound_name,'.json'],'w');
fprintf(fileID,json);
fclose(fileID);