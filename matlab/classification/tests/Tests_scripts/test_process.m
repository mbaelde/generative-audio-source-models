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
dico = 2;

if dico == 1
    T = 512*(1:5);
    nu_p = [19,9,11,11,20,11,20,10,10];
elseif dico == 2
    T = 1024*(1:5);
    nu_p = [14,5,6,7,19,6,8,6,5];
elseif dico == 3
    T = 2048*(1:5);
    nu_p = [5,3,4,4,16,3,9,4,2];
end

D = round(2^nextpow2(D_s*fs)); % Time interval between two analysys windows (sample)

N_fft = T;                     % FFT size of the analysis window
N_spect = round(N_fft/5);      % Number of points kept in the spectrum

class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
n_class = length(class);

% Parameters identification
winsize = T(1);
th_silent = -70; %dB
my_class = [length(class)+1,1:length(class)];
mu = 5;

param.winsize = winsize;
param.th_silent = th_silent;
param.N_fft = N_fft(1);
param.N_spect = N_spect(1);
param.mu = mu;
param.dico = dico;
param.nu = nu_p;
param.T = T;
%% Create test file
file_name = 'known_1';

test_sound = [];    % audio data
test_class = [];    % class indices data
test_idx = [];      % indeces of test files

for k = 1:length(class)
    folder = [data_folder,'/',class{k}];
    names = dir(folder);
    names = names(3:end);
    % Select randomly a file
    i = 1;%randi([1,length(names)]);
    % Load the file
    [aux,fs_sound] = audioread([folder,'/',names(i).name]);
    % Resample the file
    f_factor = fs / fs_sound;
    if f_factor ~= 1
        aux = resample(aux, f_factor, 1);
    end
    % Convert to mono
    if size(aux,2) == 2
        aux = mean(aux,2);
    end
    % Substract the mean
    aux = aux - mean(aux);
    % Create random silence
    silence = randi([2048,4096]);
    % Add to the test file
    test_sound = [test_sound; zeros(1024+silence,1); aux];
    test_class = [test_class; (n_class+1)*ones(1024+silence,1); k*ones(size(aux))];
    test_idx = [test_idx;i];
end
test_class = test_class';

save(['Test files/test_file_',file_name,'.mat'], 'test_sound', 'test_class', 'test_idx');

%% Edit dictionnary to remove the files in test file
load(['Test files/test_file_',file_name,'.mat'])

for dico = 1:5
    edit_dico(dico, class, T, fs, N_fft, N_spect, test_idx)
end

% for dico = 1:5;
%     disp(['--- Dico ', num2str(dico),' ---'])
%     for n = 1:length(class);
%         clear aux_L mycdf feature aux_feature
%         load(['Features/T',num2str(T(dico)),'/feature_',class{n},'.mat'])
%         
%         disp('Remove test files...')
%         cnt = 1;
%         for i = 1:length(feature)
%             if i ~= test_idx(n)
%                 aux_feature{cnt} = feature{i};
%                 cnt = cnt + 1;
%             end
%         end
%         feature = aux_feature;
%         clear aux_feature
%         
%         disp('Clear models...')
%         for cnt = 1:length(feature)
%             N_buffer = length(feature{cnt});
%             count = 1;
%             for b = 1:N_buffer
%                 if ~isempty(feature{cnt}{b}.spectrum_model)
%                     aux_feature{cnt}{count} = feature{cnt}{b};
%                     count = count + 1;
%                 end
%             end
%         end
%         feature = aux_feature;
%         
%         disp('Preprocess dictionnary...')
%         freq = 0:fs/N_fft(dico):fs/2;
%         f = freq(1:N_spect(dico)+1);
%         n_model = 10;
%         cnt_2 = 1;
%         cnt_feat = 1;
%         for feat = 1:length(feature)
%             cnt = 1;
%             N_buffer = length(feature{feat});
%             for b = 1:N_buffer
%                 mycdf(cnt_2,:) = mixture_normcdf(f, feature{feat}{b}.spectrum_model.mu, sqrt(feature{feat}{b}.spectrum_model.sigma), feature{feat}{b}.spectrum_model.mixing_coeff);
%                 if mod(cnt_2, n_model) == 0
%                     mean_cdf = mean(mycdf);
%                     aux_L{cnt_feat}(cnt,:) = [log(diff(mean_cdf)), feature{feat}{b}.class, feat, cnt];
%                     cnt_2 = 1;
%                     cnt = cnt + 1;
%                 end
%                 cnt_2 = cnt_2 + 1;
%             end
%             try aux_L{cnt_feat};
%                 cnt_feat = cnt_feat + 1;
%             catch
%                 disp('empty')
%             end
%             
%             progressbar(feat, length(feature))
%         end
%         save(['Features/T',num2str(T(dico)),'/preprocess_dictionnary_reduced_without_',class{n},'.mat'], 'aux_L')
%     end
%     
%     
%     aux_L = [];
%     for n = 1:length(class)
%         aux_L_aux = load(['Features/T',num2str(T(dico)),'/preprocess_dictionnary_reduced_without_',class{n},'.mat']);
%         aux_L = [aux_L, aux_L_aux.aux_L];
%     end
%     save(['Features/T',num2str(T(dico)),'/preprocess_dictionnary_reduced_without_T',num2str(T(dico)),'.mat'],'aux_L');
% end

%% Identification
%[L_min, L_map, L_maj] = identification_majority_vote(test_sound, my_class, param);
%[L_min, L_map] = identification_map_nu_opt(test_sound, my_class, param);
[L_min, L_map, elapsed_time_map] = identification_map(test_sound, my_class, param);
%% Plot results
figure(1)
clf
plot(test_class(T(1)/2+T(1)*(1:length(test_class)/T(1)-1)),'LineWidth',5)
hold on
plot(L_min(T(1)/2+T(1)*(1:length(L_min)/T(1)-1)), '-','LineWidth',2)
plot(L_map(T(1)/2+T(1)*(1:length(L_map)/T(1)-1)), ':','LineWidth',3)
legend('True classes', 'ML','MAP')
xlabel('Time index')
ylabel('Class')
title('Estimated class for a signal known class')

%% Compute performance measures
% for j = 1:mu
%     ML_good_classification_rate{j} = sum(test_class == L_min{j}) / length(test_class);
%     MAP_good_classification_rate{j} = sum(test_class == L_map{j}) / length(test_class);
%
%     confusion_matrix = confusionmat(test_class, L_min{j});
%     ML_confusionmatrix{j} =  confusion_matrix ./ repmat(sum( confusion_matrix,2),[1,size(confusion_matrix,2)]);
%     confusion_matrix = confusionmat(test_class, L_map{j});
%     MAP_confusionmatrix{j} =  confusion_matrix ./ repmat(sum( confusion_matrix,2),[1,size(confusion_matrix,2)]);
%
%     ML_max_buffer_detected{j} = max_number_buffer(test_class, L_min{j}, winsize, my_class);
%     MAP_max_buffer_detected{j} = max_number_buffer(test_class, L_map{j}, winsize, my_class);
%     progressbar(j,mu)
% end
% MV_good_classification_rate = sum(test_class == L_maj) / length(test_class);
% confusion_matrix = confusionmat(test_class, L_maj);
% MV_confusionmatrix =  confusion_matrix ./ repmat(sum( confusion_matrix,2),[1,size(confusion_matrix,2)]);
% MV_max_buffer_detected = max_number_buffer(test_class, L_maj, winsize, my_class);
atest_class = test_class(1:length(L_min));

ML_good_classification_rate = sum(atest_class == L_min) / length(atest_class);
MAP_good_classification_rate = sum(atest_class == L_map) / length(atest_class);

confusion_matrix = confusionmat(atest_class, L_min);
ML_confusionmatrix =  confusion_matrix ./ repmat(sum( confusion_matrix,2),[1,size(confusion_matrix,2)]);
confusion_matrix = confusionmat(atest_class, L_map);
MAP_confusionmatrix =  confusion_matrix ./ repmat(sum( confusion_matrix,2),[1,size(confusion_matrix,2)]);

ML_max_buffer_detected = max_number_buffer(test_class, L_min, winsize, my_class);
MAP_max_buffer_detected = max_number_buffer(test_class, L_map, winsize, my_class);


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