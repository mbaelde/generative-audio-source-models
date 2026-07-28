%% Script that perform learning and testing with a Coupled Matrix factorization
% method (Mesaros, 2015)
% Author: Maxime Baelde
% Date 11/05/2017
% Company: A-Volute / Inria
%% Parameters
clear

status = 'load';

addpath(genpath('Toolbox'))
addpath(genpath('Prototypes/Classification'))

param.fs = 44100;
param.T = 2048;
param.D = 512;
param.n_fft = param.T / 2 + 1;
param.power = 2;
param.threshold = -60;

n_idx = 1;
fold = 1;

data_folder = '../Data/TUT-sound-events-2017-development/';
type = 'train';
context = 'street';
file_name = [context,'_fold',num2str(fold),'_',type];
audio_folder = [data_folder,'evaluation_setup/'];

database_folder = 'Database/';
dataset_name = 'BF/';

%% Construct a database of frames from sounds in folder
if strcmp(status,'create')
    [database,activation_matrix, class, gm] = create_dataset_poly(file_name, audio_folder, param);
    if strcmp(type,'train')
        save([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'_train_',num2str(fold),'.mat'], 'database','activation_matrix','gm','class','-v7.3')
    elseif strcmp(type,'evaluate')
        save([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'_test_',num2str(fold),'.mat'], 'database','activation_matrix','gm','class','-v7.3')
    end
elseif strcmp(status,'load')
    if strcmp(dataset_name,'BF/')
        load([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'database','activation_matrix','gm','class')
    else
        load([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'_test.mat'], 'database','activation_matrix','gm','class')
    end
end
n_class = size(activation_matrix,1);

%% Construct a database of spectrogram from sounds in folder
if strcmp(status,'create')
    spectrum = compute_spectrum(database);
    save([database_folder,dataset_name,'Complete/spectrum_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'spectrum', 'activation_matrix', '-v7.3')
elseif strcmp(status,'load')
    load([database_folder,dataset_name,'Complete/spectrum_fs',num2str(param.fs),'_T',num2str(param.T),'_train.mat'], 'spectrum', 'activation_matrix')
end

%% Load predifined dataset
% Features
if strcmp(dataset_name,'BF/')
    spectrum = compute_spectrum(database);
    feature = [abs(spectrum(:,1:param.n_fft)), spectrum(:,end-1:end)];
else
    feature_training = [abs(spectrum(:,1:param.n_fft)), spectrum(:,end-1:end)];

    if strcmp(status,'create')
        save([database_folder,dataset_name,'Complete/feature_fs',num2str(param.fs),'_T',num2str(param.T),'_train.mat'], 'feature_training', 'activation_matrix', '-v7.3')
    elseif strcmp(status,'load')
        load([database_folder,dataset_name,'Complete/feature_fs',num2str(param.fs),'_T',num2str(param.T),'_train.mat'], 'feature_training', 'activation_matrix')
    end
end
clear spectrum
%%
n_idx = 1;
load([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/idx_train_test_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'idx_train', 'idx_test')

fold = 1;
id_class = unique(database(:,end-1));
n_class = length(id_class);
feature_training = [];
activation_matrix_training = [];
database_test = [];
activation_matrix_test = [];
for k = 1:n_class
    feature_class = feature(feature(:,end-1) == id_class(k),:);
    database_class = database(database(:,end-1) == id_class(k),:);
    activation_class = activation_matrix(:,database(:,end-1) == id_class(k));
    feature_training = [feature_training; feature_class(idx_train{fold}{k},:)];
    database_test = [database_test; database_class(idx_test{fold}{k},:)];
    activation_matrix_training = [activation_matrix_training, activation_class(:,idx_train{fold}{k})];
    activation_matrix_test = [activation_matrix_test, activation_class(:,idx_test{fold}{k})];
end
clear feature_class database_class activation_class
%% Learning NMF with MMLE
param.K = 10;
param.iter_max = 250;
[W,~] = learn_coupled_nmf(feature_training, activation_matrix_training, param);
%save(['Prototypes/Classification/Benchmark/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/model_matrix_fold_',num2str(fold),'.mat'],'W','H')
%% Test on mixture of dirichlet modeling
% load(['Prototypes/Classification/Benchmark/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/model_matrix_fold_',num2str(fold),'.mat'],'W')
% database_test = database;
% clear database
param.n_class = 3;

[~,C] = kmeans(W',500);
W_red = C';

method = 'ward';
W_norm = W(1:1025,:)';
W_norm = W_norm ./ repmat(sum(W_norm,2),[1,size(W_norm,2)]);
Y = pdist( sqrt(W_norm./2) );
Z = linkage(Y,method);
maxclust = 500;
idx_clusters = cluster(Z, 'maxclust', maxclust);
W_red = zeros(size(W,1),maxclust);
for nn = 1:maxclust
    W_red(:,nn) = mean(W(:,idx_clusters == nn),2);
end

% save(['Prototypes/Classification/Benchmark/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/model_matrix_fold_',num2str(fold),'.mat'],'W','W_red')

[activation_matrix_hat,computation_time] = identification_coupled_nmf(database_test(sort(randperm(6476,70)),:), W_red, param);
computation_time = computation_time(computation_time > 0);

% save(['Prototypes/Classification/Results/Benchmark/',dataset_name,'/fs',num2str(param.fs),'_T',num2str(param.T),'/result_crnn_fold_',num2str(fold),'.mat'],'activation_matrix_test','computation_time')
% load(['Prototypes/Classification/Results/Benchmark/',dataset_name,'/fs',num2str(param.fs),'_T',num2str(param.T),'/result_crnn_fold_',num2str(fold),'.mat'],'activation_matrix_test','computation_time')

%% Assign labels
m = 10;
N = size(activation_matrix_hat,2);

activation_matrix_th = [];
for b = 1:N
    if mod(b,m) == 0
        data = activation_matrix_hat(:,b-m+1:b);
        mean_value = mean(data(:));
    
        data(data >= mean_value) = 1;
        data(data < mean_value) = 0;
        activation_matrix_th = [activation_matrix_th, data];
    end
end

figure(1)
clf
imagesc(activation_matrix_test)
%
th = 0.3;
activation_matrix_th = activation_matrix_test;
activation_matrix_th(activation_matrix_th > th) = 1;
activation_matrix_th(activation_matrix_th < th) = 0;

figure(2)
clf
imagesc(activation_matrix_th)

[f1_score, error_rate] = metrics_sed(activation_matrix_test(:,1:size(activation_matrix_th,2)), activation_matrix_th)
%% Compute metrics
[f1_score, error_rate] = metrics_sed(activation_matrix(:,1:size(activation_matrix_test,2)), activation_matrix_test);

mean_time = median(computation_time);

save(['Prototypes/Classification/Results/Benchmark/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/result_coupled_nmf_fold_',num2str(fold),'.mat'],'f1_score','error_rate','mean_time')

%%
clear
class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};
n_class = length(class);

addpath('Statistics')

%% Construct learning set a-volute
fid = fopen('Polyphonic sounds/labels.txt');
sound_list = cell(1);
start_time = [];
end_time = [];
label = cell(1);

tline = fgetl(fid);
cnt = 1;
while ischar(tline)
    content = strsplit(tline,'\t');
    sound_list{cnt} = content{1};
    start_time(cnt) = str2double(content{2});
    end_time(cnt) = str2double(content{3});
    label{cnt} = content{4};
    cnt = cnt + 1;
    tline = fgetl(fid);
end
fclose(fid);

sound_name = unique(sound_list);
n_sound = length(sound_name);

V_train = [];
for n = 1:n_sound
    % read sound file
    [audio,sr] = audioread(['Polyphonic sounds/',sound_name{n}]);
    % file where the name is in lists
    cnt = 1;
    idx_name = [];
    for nn = 1:length(sound_list)
        name = sound_list{nn};
        if strcmp(name, sound_name{n})
            idx_name = [idx_name,cnt];
        end
        cnt = cnt + 1;
    end
    % convert to mono
    if size(audio,2) >= 2
        audio = mean(audio,2);
    end
    N_window = 1024;
    Window = hanning(N_window);
    Nshift = N_window / 2;
    Nfft = N_window;
    spectrogram_complex = spectrogram(audio,sqrt(Window),Nshift,Nfft);

    V_1 = abs(spectrogram_complex);
    [F,N] = size(V_1);
    % compute activation matrix based on onset and offset
    activation_matrix = zeros(n_class,N);
    start_time_idx = round(start_time(idx_name) * sr / Nshift);
    start_time_idx(start_time_idx == 0) = 1;
    end_time_idx = round(end_time(idx_name) * sr / Nshift);
    end_time_idx(end_time_idx > N) = N;
    label_name = [];
    for nn = 1:length(idx_name);
        label_name{nn} = label{idx_name(nn)};
    end
    for nn = 1:length(label_name)
        idx_class = find(ismember(class, label_name{nn}));
        activation_matrix(idx_class,start_time_idx(nn):end_time_idx(nn)) = 1;
    end
    V_2 = activation_matrix;
    aux_V = [V_1;V_2];
    %
    V_train = [V_train, aux_V];
end
N = size(V_train,2);
percent_train = 0.8;

idx_train = sort(randperm(N,round(N*percent_train)));
idx_test = setdiff(1:N,idx_train);

V_test = V_train(:,idx_test);
V_train = V_train(:,idx_train);

%%
threshold = 0:0.01:1;
f1_score = zeros(1,length(threshold));
error_rate = zeros(1,length(threshold));
for n = 1:length(threshold)
annotation_hat = W(F+1:F+E,:) * H;
annotation_hat(annotation_hat >= threshold(n)) = 1;
annotation_hat(annotation_hat < threshold(n)) = 0;

[f1_score(n), error_rate(n)] = metrics_sed(V_test(F+1:F+E,:), annotation_hat);
end

figure(3)
clf
plot(threshold,f1_score)
hold on
plot(threshold,error_rate)
legend('F1','ER')

[best_f1,idx_f1] = max(f1_score);
[best_er,idx_er] = min(error_rate);

