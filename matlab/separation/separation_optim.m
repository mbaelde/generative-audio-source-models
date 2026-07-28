%% Parameters
clear
clc
status = 'load';

addpath(genpath(pwd))

data_folder = '../Data/';
database_folder = 'Database/';
dataset_name = 'A-Volute/';

folder = [data_folder,dataset_name];

param.fs = 44100;
param.T = 2048;
param.D = 512;
param.n_fft = param.T / 2 + 1;

%% Construct a database of frames from sounds in folder
load([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'database')
load([database_folder,dataset_name,'Complete/feature_fs',num2str(param.fs),'_N',num2str(param.n_fft),'.mat'], 'feature')

n_class = max(database(:,end-1));

n_idx = 1;
load([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/idx_train_test_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'idx_train', 'idx_test')

fold = 1;
database_test = [];
feature_training = [];
for k = 1:n_class
    database_class = database(database(:,end-1) == k,:);
    database_test = [database_test; database_class(idx_test{fold}{k},:)];
    feature_class = feature(feature(:,end-1) == k,:);
    feature_training = [feature_training; feature_class(idx_train{fold}{k},:)];
end
clear database_class feature_class
clear feature database

load([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/Z_fold_',num2str(fold),'_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'Z')
N_sounds = [29   152   163    53    87]; % 96%
feature_reduced = reduce_dictionary(feature_training, Z, N_sounds);

n_models = zeros(1,n_class);
for k = 1:n_class
    n_models(k) = sum(feature_reduced(:,end-1) == k);
end

%% construct test mixture
true_class = [1,2];

prop_true = [0.5,0.5];
source = zeros(2,param.T);
mixture_sound = zeros(1,param.T);

idx = [5,10];

for k = 1:2
    data_class = database_test(database_test(:,end-1) == true_class(k),:);
    source(k,:) = data_class(idx(k),1:end-2);
    source(k,:) = source(k,:) ./ max(abs(source(k,:)));
    mixture_sound = mixture_sound + prop_true(k) * source(k,:);
end

source_spectrum = abs(fft(source,[],2));
source_spectrum = param.n_fft .* source_spectrum(:,1:param.n_fft) ./ repmat(sum(source_spectrum(:,1:param.n_fft),2),[1,param.n_fft]);

mixture_spectrum = abs(fft(mixture_sound));
mixture_spectrum = param.n_fft .* mixture_spectrum(1:param.n_fft) ./ sum(mixture_spectrum(1:param.n_fft));

idx_1 = feature_reduced(:,end-1) == true_class(1);
idx_2 = feature_reduced(:,end-1) == true_class(2);
%%
iter_nr = 10;
hat_x = rand(1,param.n_fft);
hat_x = (param.n_fft / 2) * hat_x ./ sum(hat_x);
sum_1 = sum(feature_reduced(idx_1,1:end-2));
sum_2 = sum(feature_reduced(idx_2,1:end-2));
n_1 = n_models(true_class(1));
n_2 = n_models(true_class(2));

for iter = 1:iter_nr
    % Newton Raphson
    % f_p = -n_models(true_class(1)) * psi(1, hat_x + 1) + n_models(true_class(2)) * psi(1, 2*mixture_spectrum - hat_x + 1);
    % f = -n_1 * psi(hat_x + 1) - n_2 * psi(2*mixture_spectrum - hat_x + 1) + sum_1 - sum_2;
    % hat_x = hat_x - f ./ f_p;
    
    % Newton gradient descent
    H = diag( -n_1 * psi(2, hat_x + 1) - n_2 * psi(2, 2 * mixture_spectrum - hat_x + 1) );
    grad_f = -n_1 * psi(1, hat_x + 1) + n_2 * psi(1, 2 * mixture_spectrum - hat_x + 1);
    hat_x = hat_x - (H \ grad_f')';
    
    hat_x(hat_x < 0) = 0;
end
hat_x2 = 2*mixture_spectrum - hat_x;

figure(1)
clf
plot(mixture_spectrum)
hold on
plot(0.5*hat_x)
plot(0.5*hat_x2)

figure(2)
clf
plot(source_spectrum(1,:))
hold on
plot(hat_x)

figure(3)
clf
plot(source_spectrum(2,:))
hold on
plot(hat_x2)