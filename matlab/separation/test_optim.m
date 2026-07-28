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
param.T = 512;
param.D = 512;
param.n_fft = param.T / 2 + 1;

model_folder = ['Prototypes/Classification/Benchmark/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/'];
%% Construct a database of frames from sounds in folder
load([database_folder,dataset_name,'Complete/database_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'database')
n_class = max(database(:,end-1));

n_idx = 1;
load([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/idx_train_test_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'idx_train', 'idx_test')

fold = 1;
database_test = [];
for k = 1:n_class
    database_class = database(database(:,end-1) == k,:);
    database_test = [database_test; database_class(idx_test{fold}{k},:)];
end
clear database_class

load([model_folder,'model_dmm_fold_',num2str(fold),'_nidx_',num2str(n_idx),'.mat'])

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
source_spectrum = source_spectrum(:,1:param.n_fft) ./ repmat(sum(source_spectrum(:,1:param.n_fft),2),[1,param.n_fft]);

mixture_spectrum = abs(fft(mixture_sound));
mixture_spectrum = mixture_spectrum(1:param.n_fft) ./ sum(mixture_spectrum(1:param.n_fft));

%%
M1 = length(model{true_class(1)}.prop);
M2 = length(model{true_class(2)}.prop);
hat_x1 = 2 * mixture_spectrum .* ( sum(model{true_class(1)}.alpha - 1) ) ./ ( sum(model{true_class(1)}.alpha - 1) + sum(model{true_class(2)}.alpha - 1)); 
hat_x1(hat_x1 < 0) = -hat_x1(hat_x1 < 0);
%hat_x1 = hat_x1 ./ sum(hat_x1);

hat_x2 = 2 * mixture_spectrum .* ( sum(model{true_class(2)}.alpha - 1) ) ./ ( sum(model{true_class(1)}.alpha - 1) + sum(model{true_class(2)}.alpha - 1)); 
hat_x2(hat_x2 < 0) = -hat_x2(hat_x2 < 0);
%hat_x2 = hat_x2 ./ sum(hat_x2);

figure(1)
clf
plot(hat_x1)
hold on
plot(source_spectrum(1,:))

figure(2)
clf
plot(hat_x2)
hold on
plot(source_spectrum(2,:))
%%
M1 = length(model{true_class(1)}.prop);
M2 = length(model{true_class(2)}.prop);

prop_1 = model{true_class(1)}.prop;
prop_2 = model{true_class(2)}.prop;

alpha_1 = model{true_class(1)}.alpha;
alpha_2 = model{true_class(2)}.alpha;

hat_x1 = rand(1,param.n_fft);
hat_x1 = hat_x1 ./ sum(hat_x1);

hat_x2 = rand(1,param.n_fft);
hat_x2 = hat_x2 ./ sum(hat_x2);

B_1 = gammaln(sum(alpha_1,2)) - sum(gammaln(alpha_1),2);
B_2 = gammaln(sum(alpha_2,2)) - sum(gammaln(alpha_2),2);

log_prop_1 = log(prop_1);
log_prop_2 = log(prop_2);

log_alpha_1 = log(alpha_1);
log_alpha_2 = log(alpha_2);
learning_rate = 0.9;
for iter = 1:10
    for i = 1:param.n_fft
        idx_dif_i = setdiff(1:param.n_fft,i);
        sum_diff = sum((alpha_1(:,idx_dif_i) - 1) .* repmat(log(hat_x1(idx_dif_i)),[M1,1]),2);
        F_1(i) = LSE((log_prop_1 + B_1 + log_alpha_1(:,1) + (alpha_1(:,i) - 2) .* log(hat_x1(i)) + sum_diff)');
    end
    hat_x1 = hat_x1 - learning_rate * F_1;
end