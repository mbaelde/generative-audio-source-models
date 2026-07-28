%% Script that perform learning and testing with RARE (Real-time Audio
% Recognition Engine)
% Author: Maxime Baelde
% Date 28/04/2017
% Company: A-Volute / Inria
%% Parameters
clear
clc
status = 'load';

addpath(genpath('Toolbox'))
addpath(genpath('Prototypes/Classification'))

data_folder = '../Data/';
database_folder = '../Softwares/Database/';
dataset_name = 'A-Volute/'; % or ESC-10 / ESC-50
folder = [data_folder,dataset_name];

param.fs = 44100;
param.T = 512;
param.D = 512;
param.n_fft = param.T / 2 + 1;%round(param.T / 5);
param.q = param.T / 2 + 1;
param.power = 2;
param.gpuFlag = 1;
param.threshold = -60;
param.dict = 0;
param.use_reduced = false;

%% Construct a database of frames from sounds in folder
if strcmp(status,'create')
    database = create_dataset(folder,param);
    save([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'database','-v7.3')
elseif strcmp(status,'load')
    load([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'database')
end
n_class = length(unique(database(:,end-1)));

%% Compute the Fourier spectrum of the frames in database
if strcmp(status,'create')
    spectrum = compute_spectrum(database);
    save([database_folder,dataset_name,'Complete/spectrum_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'spectrum', '-v7.3')
elseif strcmp(status,'load')
    load([database_folder,dataset_name,'Complete/spectrum_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'spectrum')
end
%% Compute the normalized energy spectra from the spectra in spectrum
if strcmp(status,'create')
    feature = compute_feature(spectrum,param);
    save([database_folder,dataset_name,'Complete/feature_fs',num2str(param.fs),'_N',num2str(param.T/2+1),'.mat'], 'feature','-v7.3')
elseif strcmp(status,'load')
    load([database_folder,dataset_name,'Complete/feature_fs',num2str(param.fs),'_N',num2str(param.T/2+1),'.mat'], 'feature')
end
clear spectrum

%% Create folds for cross-validation
for n_idx = 1:10
    disp(['n_idx:',num2str(n_idx)])
    if strcmp(status,'create')
        idx_classes = database(:,end-1);
        percent_training = 0.8;
        for n_idx = 1:10
            if ~exist([database_folder,dataset_name,'Folds/Set ',num2str(n_idx)])
                mkdir([database_folder,dataset_name,'Folds/Set ',num2str(n_idx)])
            end
            [idx_train, idx_test] = split_dataset_folds(idx_classes, percent_training);
            save([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/idx_train_test_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'idx_train', 'idx_test')
        end
    elseif strcmp(status,'load')
        load([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/idx_train_test_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'idx_train', 'idx_test')
    end
    %% Construct one fold
    for fold = 1:5
        id_class = unique(database(:,end-1));
        n_class = length(id_class);
        feature_training = [];
        database_test = [];
        %database_training = [];
        for k = 1:n_class
            feature_class = feature(feature(:,end-1) == id_class(k),:);
            database_class = database(database(:,end-1) == id_class(k),:);
            feature_training = [feature_training; feature_class(idx_train{fold}{k},:)];
            %database_training = [database_training; database_class(idx_train{fold}{k},:)];
            database_test = [database_test; database_class(idx_test{fold}{k},:)];
        end
        clear feature_class database_class
        
        %% Reduce the dictionary
        if param.use_reduced
            if strcmp(status,'create')
                Z = cluster_classes(feature_training);
                save([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/Z_fold_',num2str(fold),'_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'Z')
            elseif strcmp(status,'load')
                load([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/Z_fold_',num2str(fold),'_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'Z')
            end
  
            th.type = 'gcr';
            th.value = 99;
            th.initial_guess = [155  346  264  127  221];
            m = 39;
            N_sounds = optimize_reduce_size(feature_training, Z, database_test, m, th, param);
            % m = 39
            % N_sounds = [144  217  157   53  146]; % 99%
            % N_sounds = [154  345  263  126  220]; % 98%
            % m = 10
            % N_sounds = [29   152   163    53    87]; % 96%
            % N_sounds = [66   189   221    80   145];% 95%;
            % N_sounds = [83   227   238    97   162]; % 94.5%
            % N_sounds = [76   284   284   111   176]; %94%
            % N_sounds = [160  400   400   260   250];% 90%;
            %N_sounds = 2;
            feature_reduced = reduce_dictionary(feature_training, Z, N_sounds);
        end
        %% Identify new sounds
        id_class = setdiff(unique(database_test(:,end-1)),0);
        n_class = length(id_class);
        param.gpuFlag = 1;
        prior_g = zeros(1,n_class);
        param.use_reduced = 0;
        for k = 1:n_class
            prior_g(k) = sum(database_test(:,end-1) == id_class(k));
        end
        if param.use_reduced
            [posterior_g,computation_time] = identification(database_test, feature_reduced, prior_g, param);
        else
            [posterior_g,computation_time] = identification(database_test, feature_training, prior_g, param);
        end
        %% Assign labels
        m = 39;
        N = size(posterior_g,1);
        labels = zeros(1,N);
        for b = 1:N
            if mod(b, m) == 0
                sum_g = sum(gather(posterior_g((b-m+1):b,:)),1,'omitnan');
                if sum(sum_g) == 0
                    labels((b-m+1):b) = 0;
                else
                    labels((b-m+1):b) = find(max(sum_g) == sum_g);
                end
            end
        end
        %% Compute metrics
        true_class = database_test(:,end-1)';
        idx_zero = true_class == 0;
        true_class = true_class(~idx_zero);
        labels = labels(~idx_zero);
        
        if any(true_class == 0)
            idx_zero = labels == 0;
            true_class = true_class(~idx_zero);
            labels = labels(~idx_zero);
            confusion_matrix = confusionmat(true_class, labels, 'order', [0,n_class]);
            BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class+1]);
        else
            idx_zero = labels == 0;
            true_class = true_class(~idx_zero);
            labels = labels(~idx_zero);
            confusion_matrix = confusionmat(true_class, labels, 'order', unique(true_class));
            BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class]);
        end
        
        gcr = mean(diag(BAY_confusionmatrix))*100;
        
        mean_time = median(computation_time);
        %%
        if param.use_reduced
            save(['Prototypes/Classification/Results/Reduced dictionary/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/result_fold_',num2str(fold),'_nidx_',num2str(n_idx),'_nsound_',num2str(N_sounds),'.mat'],'gcr','mean_time')
        else
            save(['Prototypes/Classification/Results/Full dictionary/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/result_fold_',num2str(fold),'_nidx_',num2str(n_idx),'.mat'],'gcr','mean_time')
        end
    end
end
