clear
addpath(genpath('Toolbox'))
addpath(genpath('Prototypes'))

% load data
load('F:\Ph. D Thesis\Softwares\Database\A-Volute\Complete\database_fs44100_T1024.mat')
load('F:\Ph. D Thesis\Softwares\Database\A-Volute\Complete\feature_fs44100_N513.mat')
T = size(database,2)-2;
F = size(feature,2)-2;

n_idx = 1;
fold = 1;
load(['F:\Ph. D Thesis\Softwares\Database\A-Volute\Folds\Set ',num2str(n_idx),'\idx_train_test_fs44100_T2048.mat'], 'idx_train', 'idx_test')
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
clear feature_class database_class feature database
load(['F:\Ph. D Thesis\Softwares\Database\A-Volute\Folds\Set ',num2str(n_idx),'\Z_fold_',num2str(fold),'_fs44100_T2048.mat'], 'Z')
N_sounds = [144  217  157   53  146]; % 99%
feature = reduce_dictionary(feature_training, Z, N_sounds);

% get classes
idx_class  = feature(:,end-1);
n_class = length(unique(idx_class));

% get number of models per classes
n_per_class = zeros(1,n_class);
for k = 1:n_class
    n_per_class(k) = sum(idx_class == k);
end
cn_per_class = cumsum([0,n_per_class]);
%%
p = cell(1,n_class);
M = zeros(1,n_class);
for k = 1:n_class
    p{k} = feature(cn_per_class(k)+1:cn_per_class(k+1),1:end-2)';
    M(k) = size(p{k},2);
end

n_comb_2 = nchoosek(n_class,2);
gm = zeros(n_comb_2,2);
cnt = 1;
for k = 1:n_class
    for m = k+1:n_class
        gm(cnt,:) = [k,m];
        cnt = cnt + 1;
    end
end

%% Two source mixture
q = size(p_1,1);
% sample from the models
x_1 = mixture_mnrnd(1, q, p{2}, 1./ones(M(1),1))' + eps;
x_2 = mixture_mnrnd(1, q, p{5}, 1./ones(M(2),1))' + eps;

y = x_1 + x_2;

figure(2)
clf
plot(x_1)
hold on
plot(x_2)
plot(y)
legend('x_1','x_2','y')
% compute distribution
mask = zeros(F,n_class);
likelihood_y = zeros(1,n_class);
for k = 1:n_comb_2
    tic
    [mask(:,k),likelihood_y(k)] = compute_masking_mex(y, p{gm(k,1)}, p{gm(k,2)});
    toc
end
best_idx = find(likelihood_y == max(likelihood_y))

hat_x_1 = y .* mask(:,best_idx);
hat_x_2 = y .* (1-mask(:,best_idx));

figure(3)
clf
plot(hat_x_1)
hold on
plot(hat_x_2)
title('Estimated sources')

figure(4)
clf
plot(x_1)
hold on
plot(hat_x_1)
title('Difference between true and estimated x_1')

figure(5)
clf
plot(x_2)
hold on
plot(hat_x_2)
title('Difference between true and estimated x_2')

%%
mu_1 = q * p_1;
mu_2 = q * p_2;
sigma_1 = zeros(F,F,M_1);
sigma_2 = zeros(F,F,M_2);
for i = 1:M_1
    sigma_1(:,:,i) = q * (diag(p_1(:,i)) - p_1(:,i) * p_1(:,i)');
end
for i = 1:M_2
    sigma_2(:,:,i) = q * (diag(p_2(:,i)) - p_2(:,i) * p_2(:,i)');
end

prob_psi = zeros(M_1,M_2);
for j = 1:M_1
    for k = 1:M_2
        sum_mu = mu_1(:,j) + mu_2(:,k);
        sum_sigma = sigma_1(:,:,j) + sigma_2(:,:,k) + eps;
        
        prob_psi(j,k) = 1 / mvnpdf(y, sum_mu, sum_sigma);
    end
end

hat_x_1 = zeros(F,1);
hat_x_2 = zeros(F,1);
for j = 1:M_1
    for k = 1:M_2
        hat_x_1 = hat_x_1 + mu_1(:,j) ./ (norm_factor(j,k)+eps);
        hat_x_2 = hat_x_2 + mu_2(:,j) ./ (norm_factor(j,k)+eps);
    end
end


%%

x_1 = mnrnd(q,p_1);
x_2 = mnrnd(q,p_2);

y = x_1 + x_2;

figure(2)
clf
plot(x_1)
hold on
plot(x_2)
plot(y)

n_samples = 1000;
x_1samples = zeros(n_samples,F);
for nn = 1:n_samples
    x_1samples(nn,:) = binornd(y, p_1./(p_1+p_2));
end
x_1samples = F * x_1samples ./ repmat(sum(x_1samples,2),[1,F]);

x_1hat = median(x_1samples);
x_2hat = y - x_1hat;

figure(3)
clf
plot(x_1hat)
hold on
plot(x_2hat)