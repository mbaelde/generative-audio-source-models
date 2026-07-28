clear
clc
data_folder = '../../Data/';
addpath(genpath('AudioDescriptors'))
addpath(genpath('Statistics'))
addpath(genpath('Features'))
addpath(genpath('Results'))
warning off

%% Initialisation
fs = 44100;                    % Sampling rate
D_s = 0.01;                    % Time interval between two analysys windows (sec)

T = [512, 1024, 2048];
D = round(2^nextpow2(D_s*fs)); % Time interval between two analysys windows (sample)

N_fft = T;                     % FFT size of the analysis window
N_spect = round(N_fft/5);      % Number of points kept in the spectrum

class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
n_class = length(class);

folder_feat = 'Clean database features/';
folder_complex = 'Complex/[Save]/';

%% Initialise
my_class = 1:length(class);

dico = 2;
f = 0:fs/N_fft(dico):fs/2;
f = f(1:N_spect(dico));

disp(['---- Dico : ', num2str(dico),' ----'])
% Load preprocess dictionnary
% aux_feature = [];
% for k = 1:n_class
%     load(['Features/',folder_feat,folder_complex,'T',num2str(T(dico)),'/feature_',class{k},'.mat'])
%     aux_feature = [aux_feature, feature];
% end
% feature = aux_feature;
% clear aux_feature
% load(['Features/',folder_feat,folder_complex,'T',num2str(T(dico)),'/id_error.mat'])
% feature = [];
% cnt = 1;
% idx_sound_1 = [];
% for i = 1:length(aux_feature)
%     if ~isempty(aux_feature{i})
%         feature{cnt} = aux_feature{i};
%         idx_sound_1 = [idx_sound_1,i];
%         cnt = cnt + 1;
%     end
% end
% aux_feature = [];
% cnt = 1;
% idx_sound_2 = [];
% for i = 1:length(feature)
%     if ~any(i == id_error{1})
%         aux_feature{cnt} = feature{i};
%         idx_sound_2 = [idx_sound_2,i];
%         cnt = cnt + 1;
%     end
% end

%%
fold = 1;
winsize = T(dico);

param.N_spect = N_spect(dico);
param.winsize = winsize;
param.fs = fs;
param.type = '';
% param.idx_sound_1 = idx_sound_1;
% param.idx_sound_2 = idx_sound_2;
param.f = f;
% load(['Results/Performance/Complex/dataset_T',num2str(T(dico)),'_all.mat'])
% param.model_size = model_size;
% 
% database = database_test;
% feature = feature_training;
% clear feature_test feature_training database_test database_training
load('Database/T1024/On Models/Complex/dataset_T1024_discr_fold_1.mat'])


% if dico == 1
%     load(['Database/T',num2str(T(dico)),'/database.mat'])
% else
%     load(['Database/Overlap/T',num2str(T(dico)),'/database.mat'])
% end
% 
% feature_training = [];
% feature_test = [];
% percent_test = 0.20;
% database_training = [];
% database_test = [];
% max_old = 0;
% model_size = zeros(1,n_class);
% cnt = 1;
% cnt_t = 1;
% for k = 2:n_class
%     load(['Features/',folder_feat,folder_complex,'T',num2str(T(dico)),'/feature_',class{k},'.mat'])
%     n_samp = length(feature);
%     n_test = floor(n_samp * percent_test);
%     idx_test = sort(randperm(n_samp, n_test));
%     idx_training = setdiff(1:n_samp, idx_test);
%     database_training = [database_training; database(idx_training+max_old,:)];
%     database_test = [database_test; database(idx_test+max_old,:)];
%     for ii = 1:n_samp
%         if all(ii~=idx_test)
%             feature_training{cnt} = feature{ii};
%             cnt = cnt + 1;
%         else 
%             feature_test{cnt_t} = feature{ii};
%             cnt_t = cnt_t + 1;
%         end
%     end
%     model_size(k) = n_samp - length(idx_test);
%     max_old = max_old + n_samp;
% end


%load(['Features/',folder_feat,folder_complex,'T',num2str(T(dico)),'/feature_T',num2str(T(dico)),'.mat'])
%
%save(['Results/Performance/Complex/dataset_T',num2str(T(dico)),'_all.mat'],'database_training', 'database_test','feature_training', 'feature_test', 'model_size')

tic
[L_min, L_map, L_bay] = identification_complex(database_test, feature_training, my_class, param);
elapsed_time = toc;

L_min = L_min(L_min > 0);
L_map = L_map(L_map > 0);
L_bay = L_bay(L_bay > 0);

n_max = length(L_min);

N = size(database,1);

confusion_matrix = confusionmat(database_test(1:n_max,end-1), L_min, 'order', 1:9);
confusion_matrix = confusion_matrix;
ML_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

ML_good_classification_rate = diag(ML_confusionmatrix);

confusion_matrix = confusionmat(database_test(1:n_max,end-1), L_map, 'order', 1:9);
confusion_matrix = confusion_matrix;
MAP_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

MAP_good_classification_rate = diag(MAP_confusionmatrix);

confusion_matrix = confusionmat(database_test(1:n_max,end-1), L_bay, 'order', 1:9);
confusion_matrix = confusion_matrix;
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

BAY_good_classification_rate = diag(BAY_confusionmatrix);

%
%for k = 1:length(class)
%    ML_gcr_per_class(k) = mean(ML_good_classification_rate{k});
%    MAP_gcr_per_class(k) = mean(MAP_good_classification_rate{k});
%    
%    ML_confmat_per_class(k,:) = mean(ML_confusionmatrix{k});
%    MAP_confmat_per_class(k,:) = mean(MAP_confusionmatrix{k});
%    
 %   ML_max_buffer_per_class(k,:) = max(ML_max_buffer_detected{k});
 %   MAP_max_buffer_per_class(k,:) = max(MAP_max_buffer_detected{k});
% end
%
save(['Results/Performance/Complex/result_dataset_T',num2str(T(dico)),'_all.mat'], 'L_min', 'L_map', 'L_bay', 'ML_confusionmatrix', 'MAP_confusionmatrix', 'BAY_confusionmatrix', 'elapsed_time')

