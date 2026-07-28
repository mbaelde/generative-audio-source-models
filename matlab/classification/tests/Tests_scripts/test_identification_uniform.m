clear
clc
data_folder = '../../Data/';
folder_database = 'A-Volute/';
addpath(genpath('AudioDescriptors'))
addpath(genpath('Statistics'))
addpath(genpath('Features'))
addpath(genpath('Results'))
addpath(genpath('Database'))
addpath(genpath('Identification procedure'))
warning off
distcomp.feature( 'LocalUseMpiexec', false )
%% Initialisation
dico = 3;

fs = [11025,22050,44100];                    % Sampling rate
D_s = 0.01;                    % Time interval between two analysys windows (sec)

T = [512, 1024, 2048, 4096];
D = round(2^nextpow2(D_s*fs(dico))); % Time interval between two analysys windows (sample)

N_fft = T;                     % FFT size of the analysis window
N_spect = T/2+1;%round(N_fft/5);      % Number of points kept in the spectrum

my_names = dir([data_folder,folder_database]);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end
class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};
n_class = length(class);

%% Initialise
my_class = 1:length(class);

type_data = 'Models';
disp(['---- Dico : ', num2str(dico),' ----'])

winsize = T(dico);

f = 0:fs/N_fft(dico):fs/2;
f = f(1:N_spect(dico));

nn = 3;
param.N_spect = N_spect(nn);
param.winsize = winsize;
param.fs = fs(dico);
param.type = 'clavel';
param.f = f;
type = '80-20/';
%%
if strcmp(type,'50-50/')
    num_fold = 2;
else
    num_fold = 5;
end

for fold = 1:num_fold
disp(['-- Fold : ', num2str(fold),' --'])
if n_class == 5
    %load(['Database/',folder_database,'FS',num2str(fs(dico)),'/T',num2str(T(dico)),'/Metaclasse/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'],'database_test','feature_training')
    load(['Database/',folder_database,'FS',num2str(fs(dico)),'/T',num2str(T(dico)),'/Metaclasse/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'],'database_test','raw_spectrum_training')
    feature_training = [abs(raw_spectrum_training(:,1:N_spect(dico))).^2, raw_spectrum_training(:,end-1:end)];
    clear raw_spectrum_training
else
    load(['Database/',folder_database,'FS',num2str(fs(dico)),'/T',num2str(T(dico)),'/Uniform/',type,'dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'])
end
if ~exist('feature_training')
    feature_training = aux_L_training;
    feature_test = aux_L_test;
    clear aux_L_training aux_L_test
else
    % Preprocess the dictionary
    feature_training(:,1:end-2) = feature_training(:,1:end-2) ./ repmat(sum(feature_training(:,1:end-2),2),[1,N_spect(dico)]);
    feature_training(:,1:end-2) = log(feature_training(:,1:end-2));
end
clear feature_test database_training
param.snr = inf;
param.gpuFlag = 1;

for k = 1:n_class
    prior_g(k) = sum(database_test(:,end-1) ==k);
end

N = size(database_test,1);

n_buff = 1;

param.n_buff = n_buff;
param.dict = 0;

[~,posterior_g,computation_time] = identification_general(database_test, feature_training, prior_g, my_class, param);

% idx_true = sort(randperm(17748,400));
% N = length(idx_true);
% [~,posterior_g] = identification_noyau(database_test, feature_training, prior_g, my_class, param);
%[~,posterior_g] = identification_dirichlet(database_test, feature_training, prior_g, my_class, param);

for n_buff = 1:20
    L_bay = zeros(1,N);
    disp(['n_buff=',num2str(n_buff)])
    for b = 1:N
        if mod(b, n_buff) == 0
            sum_g = sum(gather(posterior_g((b-n_buff+1):b,:)),1,'omitnan');
            L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
        end
    end
    idx_p = L_bay > 0;
    L_bay = L_bay(idx_p);
    true_class = database_test(:,end-1)';
    %true_class = database_test(idx_true,end-1)';
    true_class = true_class(idx_p);
    confusion_matrix = confusionmat(true_class, L_bay, 'order', unique(true_class));
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class]);
    gcr(n_buff) = mean(diag(BAY_confusionmatrix))*100;
    %save(['Results/Performance/N_spect/',num2str(param.N_spect),'/',folder_database,'result_dataset_',num2str(dico),'_fold_',num2str(fold),'_nbuff_',num2str(n_buff),'.mat'], 'L_bay', 'BAY_confusionmatrix')
    %save(['Results/Performance/Temporal horizon/',folder_database,'FS',num2str(fs(dico)),'/',type,'result_dataset_',num2str(dico),'_fold_',num2str(fold),'_nbuff_',num2str(n_buff),'.mat'], 'L_bay', 'BAY_confusionmatrix')
    %save(['Results/Performance/',folder_database,'result_dataset_',num2str(dico),'_fold_',num2str(num_base),'_nbuff_',num2str(n_buff),'.mat'], 'L_bay', 'BAY_confusionmatrix')
    save(['Results/Papier Pattern Recognition/',folder_database,'Fold ',num2str(fold),'/result_dataset_',num2str(dico),'_nbuff_',num2str(n_buff),'.mat'], 'L_bay', 'BAY_confusionmatrix', 'computation_time')

end

end

%%
gm = [1,1;2,2;3,3;4,4;5,5];
matrix_true = zeros(n_class, length(L_bay));
matrix_pred = zeros(n_class, length(L_bay));
for n = 1:length(L_bay)
    matrix_true(unique(gm(true_class(n),:)),n) = 1;
    matrix_pred(unique(gm(L_bay(n),:)),n) = 1;
end

% Segment-based metrics
true_positive = 0;
false_positive = 0;
false_negative = 0;
for n = 1:length(L_bay)
    true = matrix_true(:,n);
    pred = matrix_pred(:,n);
    for k = 1:n_class
        if true(k) == 1 && pred(k) == 1
            true_positive = true_positive + 1;
        elseif true(k) == 1 && pred(k) == 0
            false_negative = false_negative + 1;
        elseif true(k) == 0 && pred(k) == 1;
            false_positive = false_positive + 1;
        end
    end
end
precision = true_positive ./ (true_positive + false_positive);
recall = true_positive ./ (true_positive + false_negative);

f1_score = 2 * precision * recall ./ (precision + recall);