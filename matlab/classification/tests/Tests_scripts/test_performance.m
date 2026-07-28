 clear
clc
data_folder = '../../Data/';
folder_esc = 'ESC-50/';
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

my_names = dir([data_folder,folder_esc]);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end

%class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
n_class = length(class);

folder_feat = 'Clean database features/';
folder_interval = 'Interval/';
type = '';

%% Initialise
my_class = 1:length(class);

folder_save = 'Folds 80 train 20 test/';
dico = 3;
num_base = 2;
type_data = 'Models';
disp(['---- Dico : ', num2str(dico),' ----'])
disp(['-- Fold : ', num2str(num_base),' --'])

if num_base == 0
    % Load preprocess dictionnary
    if strcmp(folder_interval(1:3), 'Non')
        load(['Features/',folder_feat,folder_interval,'T',num2str(T(dico)),'/feature_T',num2str(T(dico)),'_clear.mat']);
    else
        %load(['Features/',folder_feat,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_reduced_T',num2str(T(dico)),'.mat']);
        load(['Features/',folder_feat,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_T',num2str(T(dico)),'.mat']);
    end
end

winsize = T(dico);

f = 0:fs/N_fft(dico):fs/2;
f = f(1:N_spect(dico));

param.N_spect = N_spect(dico);
param.winsize = winsize;
param.fs = fs;
param.type = 'clavel';
param.f = f;

if num_base == 0
    if dico == 1
        load(['Database/T',num2str(T(dico)),'/database.mat'])
    else
        load(['Database/T',num2str(T(dico)),'/database_overlap.mat'])
    end
else
    if strcmp(type,'Complex')
        load(['Database/T',num2str(T(dico)),'/On ',type_data,'/Complex/dataset_T',num2str(T(dico)),'_fold_',num2str(num_base),'.mat'])
    else
        %load(['Database/T',num2str(T(dico)),'/On ',type_data,'/dataset_T',num2str(T(dico)),'_fold_',num2str(num_base),'.mat'])
        load(['Database/',folder_esc,'T',num2str(T(dico)),'/On ',type_data,'/dataset_T',num2str(T(dico)),'_fold_',num2str(num_base),'.mat'])
    end
    if strcmp(type,'Reduce')
        clear aux_L_training aux_L_test database_training
        load(['Features\Clean database features\Interval\T',num2str(T(dico)),'\reduce_dict.mat'])
    end
end
% percent_training = 0.8;
% [aux_L_training, aux_L_test, database_training, database_test] = split_dataset(aux_L, database, percent_training);
% save(['Results/Performance/dataset_',num2str(dico),'_3.mat'],'aux_L_training', 'aux_L_test','database_training','database_test')
%database = database(1:200,:);

% [~, ~, ~, database_test] = split_dataset(aux_L, database, 0.99);
% database = database_test;

param.snr = inf;
%%
%for n_class = 2:10
if num_base == 0
    [L_min, L_map, L_bay] = identification_general(database, aux_L, my_class, param);
else
    if strcmp(type,'Complex')
        L_bay = identification_complex_discr(database_test, aux_L_training, my_class, param);
    elseif strcmp(type,'Reduce')
        L_bay = identification_reduce(database_test, dictionary(:,1:end-1), my_class, param);
    else

        %idx_class = aux_L_training(:,end-1) < n_class + 1;
        for k = 1:n_class
            prior_g(k) = sum(idx_test{k});
            %prior_g(k) = sum(database_test(:,end-1) ==k);
        end
        for k = 1:n_class
            disp(['Currently: class ', class{k}])
            load(['Database/',folder_esc,'T',num2str(T(dico)),'/database_overlap_',class{k},'.mat'])
            database_test = database(idx_test{k},:);
            tic
            L_bay{k} = identification_general(database_test, aux_L_training, prior_g, my_class(1:n_class), param);
            elapsed_time{k} = toc / prior_g(k);
            save('tmp_result.mat','L_bay','elapsed_time')
        end
    end
end

aux_L_bay = [];
for k = 1:n_class
aux_L_bay = [aux_L_bay, L_bay{k}];
end
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

%time(n_class) = mean(cell2mat(elapsed_time));
%err(n_class) = 1 - mean(diag(BAY_confusionmatrix));

%save('Results/Performance/comp_per_nb_class.mat','time','err')
%end
%%
L_min = L_min(L_min > 0);
L_map = L_map(L_map > 0);
L_bay = L_bay(L_bay > 0);

n_max = length(L_bay);

N = size(database,1);

if num_base == 0
    confusion_matrix = confusionmat(database(idx_min,end-1), L_min, 'order', 1:9);
    ML_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

    ML_good_classification_rate = diag(ML_confusionmatrix);

    confusion_matrix = confusionmat(database(idx_map,end-1), L_map, 'order', 1:9);
    MAP_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

    MAP_good_classification_rate = diag(MAP_confusionmatrix);

    confusion_matrix = confusionmat(database_test(1:n_max,end-1), L_bay, 'order', 1:9);
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

    BAY_good_classification_rate = diag(BAY_confusionmatrix);
else
    confusion_matrix = confusionmat(database_test(1:n_max,end-1), L_min, 'order', 1:9);
    ML_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

    ML_good_classification_rate = diag(ML_confusionmatrix);

    confusion_matrix = confusionmat(database_test(1:n_max,end-1), L_map, 'order', 1:9);
    MAP_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

    MAP_good_classification_rate = diag(MAP_confusionmatrix);

    confusion_matrix = confusionmat(database_test(1:n_max,end-1), L_bay, 'order', 1:9);
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,length(class)]);

    BAY_good_classification_rate = diag(BAY_confusionmatrix);
end
%
if num_base == 0
    if strcmp(param.type,'')
        save(['Results/Performance/New/result_dico_T',num2str(T(dico)),'.mat'], 'L_min', 'L_map', 'L_bay', 'ML_confusionmatrix', 'MAP_confusionmatrix', 'BAY_confusionmatrix')
    elseif strcmp(param.type,'clavel')
        save(['Results/Performance/Comp Clavel/result_dico_T',num2str(T(dico)),'_clavel.mat'], 'L_min', 'L_map', 'L_bay', 'ML_confusionmatrix', 'MAP_confusionmatrix', 'BAY_confusionmatrix')
    end
else
    if strcmp(param.type,'')
        save(['Results/Performance/Complex/result_dataset_discr_',type_data,'_',num2str(dico),'_fold_',num2str(num_base),'.mat'], 'L_bay','BAY_confusionmatrix')
    elseif strcmp(param.type,'clavel')
        save(['Results/Performance/Comp Clavel/', folder_save, 'SNR/result_dataset_',type_data,'_',num2str(dico),'_fold_',num2str(num_base),'_snr_',num2str(param.snr),'.mat'], 'L_min', 'L_map', 'L_bay', 'ML_confusionmatrix', 'MAP_confusionmatrix', 'BAY_confusionmatrix')
    end
end
%% Identification
winsize = 512;
th_silent = -70; %dB
my_class = [length(class)+1,1:length(class)];

param.winsize = winsize;
param.th_silent = th_silent;
param.N_fft = N_fft;
param.N_spect = N_spect;

[L_min, L_map, L_maj] = identification_majority_vote(test_sound, my_class, mu, aux_L, model_size, param);

figure(1)
clf
plot(test_class,'LineWidth',5)
hold on
plot(L_maj, ':','LineWidth',5)
legend('True classes', 'Majority vote')
xlabel('Time index')
ylabel('Class')
title('Estimated class for a signal known class')

%% Compute performance measures
for j = 1:mu
    ML_good_classification_rate{j} = sum(test_class == L_min{j}) / length(test_class);
    MAP_good_classification_rate{j} = sum(test_class == L_map{j}) / length(test_class);
    
    confusion_matrix = confusionmat(test_class, L_min{j});
    ML_confusionmatrix{j} =  confusion_matrix ./ repmat(sum( confusion_matrix,2),[1,size(confusion_matrix,2)]);
    confusion_matrix = confusionmat(test_class, L_map{j});
    MAP_confusionmatrix{j} =  confusion_matrix ./ repmat(sum( confusion_matrix,2),[1,size(confusion_matrix,2)]);
    
    ML_max_buffer_detected{j} = max_number_buffer(test_class, L_min{j}, winsize, my_class);
    MAP_max_buffer_detected{j} = max_number_buffer(test_class, L_map{j}, winsize, my_class);
    progressbar(j,mu)
end
MV_good_classification_rate = sum(test_class == L_maj) / length(test_class);
confusion_matrix = confusionmat(test_class, L_maj);
MV_confusionmatrix =  confusion_matrix ./ repmat(sum( confusion_matrix,2),[1,size(confusion_matrix,2)]);
MV_max_buffer_detected = max_number_buffer(test_class, L_maj, winsize, my_class);

%% Store results
result.L_min = L_min;
result.L_map = L_map;
result.L_maj = L_maj;
result.ML_good_classification_rate = ML_good_classification_rate;
result.MAP_good_classification_rate = MAP_good_classification_rate;
result.MV_good_classification_rate = MV_good_classification_rate;
result.ML_confusionmatrix = ML_confusionmatrix;
result.MAP_confusionmatrix = MAP_confusionmatrix;
result.MV_confusionmatrix = MV_confusionmatrix;
result.ML_max_buffer_detected = ML_max_buffer_detected;
result.MAP_max_buffer_detected = MAP_max_buffer_detected;
result.MV_max_buffer_detected = MV_max_buffer_detected;

if ~exist(['Results/Majority vote with indecision reduced dictionnary/Son ',file_name,'/'])
    mkdir(['Results/Majority vote with indecision reduced dictionnary/Son ',file_name,'/'])
end
save(['Results/Majority vote with indecision reduced dictionnary/Son ',file_name,'/result_buffer_nu',num2str(nu),'.mat'],'result')