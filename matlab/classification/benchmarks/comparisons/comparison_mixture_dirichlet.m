%% Script that perform learning and testing with a Dirichlet Mixture Model
% Author: Maxime Baelde
% Date 04/05/2017
% Company: A-Volute / Inria
%% Parameters
clear

status = 'load';

addpath(genpath('Toolbox'))
addpath(genpath('Prototypes/Classification'))

data_folder = '../Data/';
database_folder = '../Softwares/Database/';
dataset_name = 'ESC-10/';
folder = [data_folder,dataset_name];

param.fs = 44100;
param.T = 2048;
param.D = 512;
param.n_fft = param.T / 2 + 1;
param.power = 2;
param.threshold = -60;

param.learning_rate = 0.9;
param.M_max = [5,10,20,30];
param.try_max = 10;
param.type = 'newton';
param.iter_max = 50;
param.iter_nr_max = 20;

% distcomp.feature( 'LocalUseMpiexec', false )
% gcp
%% Load predifined dataset
% Database
database = load([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'database');
database = database.database;
n_class = max(database(:,end-1));

% Features
load([database_folder,dataset_name,'Complete/feature_fs',num2str(param.fs),'_N',num2str(param.n_fft),'.mat'], 'feature')
%%
for n_idx = 1:10
% Folds
load([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/idx_train_test_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'idx_train', 'idx_test')
%%
for fold = 1:5
feature_training = [];
database_test = [];
for k = 1:n_class
    feature_class = feature(feature(:,end-1) == k,:);
    database_class = database(database(:,end-1) == k,:);
    feature_training = [feature_training; feature_class(idx_train{fold}{k},:)];
    database_test = [database_test; database_class(idx_test{fold}{k},:)];
end
clear feature_class database_class
param.n_idx = n_idx;
param.fold = fold;
%% Learning DMM and choose optimal number of components
model = learn_dmm(feature_training, param);
save(['Prototypes/Classification/Benchmark/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/model_dmm_fold_',num2str(fold),'_nidx_',num2str(n_idx),'.mat'],'model')
%% Test on mixture of dirichlet modeling
[posterior_g,computation_time] = identification_dmm(database_test, model, param);

%% Assign labels
m = 10;
N = size(posterior_g,1);
labels = zeros(1,N);
for b = 1:N
    if mod(b, m) == 0
        sum_g = sum(gather(posterior_g((b-m+1):b,:)),1,'omitnan');
        labels((b-m+1):b) = find(max(sum_g) == sum_g);
    end
end
%% Compute metrics
true_class = database_test(:,end-1)';

if any(true_class == 0)
    confusion_matrix = confusionmat(true_class, labels, 'order', [0,unique(true_class)]);
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class+1]);
else
    idx_zero = labels == 0;
    true_class = true_class(~idx_zero);
    labels = labels(~idx_zero);
    confusion_matrix = confusionmat(true_class, labels, 'order', [unique(true_class)]);
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class]);
end

gcr = mean(diag(BAY_confusionmatrix))*100;
mean_time = median(computation_time);
save(['Prototypes/Classification/Results/Benchmark/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/result_dmm_fold_',num2str(fold),'_nidx_',num2str(n_idx),'.mat'],'gcr','mean_time')
end
end