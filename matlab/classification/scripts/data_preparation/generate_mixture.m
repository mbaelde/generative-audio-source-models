clear
clc

data_folder = '../../Data/';
database_folder = 'A-Volute/';

startup
addpath(genpath('Database'))
addpath(genpath('Statistics'))
addpath(genpath('Tree functions'))
addpath(genpath('Identification procedure'))
addpath(genpath('Dictionary creation'))
%distcomp.feature( 'LocalUseMpiexec', false )
%gcp
%% Initialisation
dico = 3;
fold = 1;

%% Load data
load(['Database/',database_folder,'FS',num2str(fs),'/T',num2str(T(dico)),'/Uniform/80-20/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'])
clear database_training feature_test

%% Enumerate the possibilities
gm = [];
cnt = 1;
for k = 1:n_class
    for m = k+1:n_class
        gm(cnt,:) = [k,m];
        cnt = cnt + 1;
    end
end
n_comb = size(gm,1);
%% Create complete mixture dictionary based on reduced dictionary
N_sounds = [91, 69, 53, 70, 70, 85, 105, 62, 100];
feature_norm = create_reduced_dictionary(feature_training, N_sounds, database_folder, fold, 'single');

g = gm(26,:);

n_models = zeros(1,n_class);
for k = 1:n_class
    n_models(k) = sum(feature_norm(:,end-1) == k);
end

n_prod = zeros(1,n_comb);
for k = 1:n_comb
    n_prod(k) = prod(n_models(gm(k,:)));
end

n_mixture = sumn_prod(26);
%%
%mixture_dictionary = zeros(n_mixture, size(feature_norm,2));
%cnt = 0;
n_cluster = zeros(1,n_class);
idx_clusters = cell(1,n_class);
N_sounds_mixture = 100;
mixture_reduced = [];
for k = 1:n_comb
    g = gm(k,:);
    feature_1 = feature_norm(feature_norm(:,end-1) == g(1),:);
    feature_2 = feature_norm(feature_norm(:,end-1) == g(2),:);
    
    aux = zeros(n_prod(k),N_spect(dico));
    for n = 1:n_models(g(1))
        aux((n-1)*n_models(g(2))+1:n*n_models(g(2)),:) = 0.5 * (repmat(feature_1(n,1:end-2), [n_models(g(2)),1]) + feature_2(:,1:end-2));
    end
%     mixture_dictionary(cnt+1:cnt+n_prod(k),:) = [aux, k * ones(n_prod(k),1),1*ones(n_prod(k),1)];
%     cnt = cnt + n_prod(k);
    clear feature_1 feature_2
    
    % Reduce mixture dictionary
    feature_class = aux;
    n_cluster(k) = floor(size(feature_class,1) / N_sounds_mixture);
    %feature_class = mixture_dictionary(mixture_dictionary(:,end-1) == k,:);
    Y = pdist(sqrt(feature_class/2));
    method = 'ward';
    Z = linkage(Y,method);
    %save(['Results/Mixture/Hierarchical/',database_folder,'Mixtures/Fold ',num2str(fold),'/Z_',num2str(k),'.mat'],'Z')
    idx_clusters{k} = cluster(Z, 'maxclust', n_cluster(k));
    
    my_feature = zeros(n_cluster(k),N_spect(dico)+2);
    for nn = 1:n_cluster(k)
        my_feature(nn,:) = [mean(feature_class(idx_clusters{k} == nn,:),1),k,1];
    end
    my_feature(:,1:end-2) = my_feature(:,1:end-2) ./ repmat(sum(my_feature(:,1:end-2),2),[1,N_spect(dico)]);
    
    mixture_reduced = [mixture_reduced; my_feature];
    progressbar(k,n_comb)
end

%mixture_reduced = create_reduced_dictionary(mixture_dictionary, N_sounds_mixture, database_folder, fold, 'mixture');
mixture_reduced(:,end-1) = mixture_reduced(:,end-1) + n_class;
clear mixture_dictionary
%% Complete dictionary
dictionary = [feature_norm; mixture_reduced];
aux_L = dictionary;
aux_L(:,1:end-2) = log(aux_L(:,1:end-2));

%% Test on learning set
true_class = dictionary(:,end-1);

prior_g = zeros(1,n_class + n_comb);
for k = 1:(n_class + n_comb)
    prior_g(k) = sum(dictionary(:,end-1) == k);
end
my_class = 1:(n_class + n_comb);
param.N_spect = N_spect(dico);
param.gpuFlag = 0;
n_buff = 1;
param.n_buff = n_buff;

tic
L_bay = identification_general(dictionary, aux_L, prior_g, my_class, param);
elapsed_time = toc;

idx_p = L_bay > 0;
L_bay = L_bay(idx_p);
true_class = true_class(idx_p);

confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class+n_comb]);
mean(diag(BAY_confusionmatrix))
elapsed_time / size(database_test,1)

%% Test on test set
% Create test set
aux_test = database_test(sort(randperm(size(database_test,1),floor(size(database_test,1)/50))),:);

n_test = zeros(1,n_class);
for k = 1:n_class
    n_test(k) = sum(aux_test(:,end-1) == k);
end

n_prod = zeros(1,n_comb);
for k = 1:n_comb
    n_prod(k) = prod(n_test(gm(k,:)));
end
n_test_mixt = sum(n_prod);

test_dictionary = zeros(n_test_mixt, size(aux_test,2));
cnt = 0;
for k = 1:n_comb
    g = gm(k,:);
    test_1 = aux_test(aux_test(:,end-1) == g(1),:);
    test_1(:,1:end-2) = test_1(:,1:end-2) ./ repmat(max(abs(test_1(:,1:end-2)),[],2),[1,T(dico)]);
    test_2 = aux_test(aux_test(:,end-1) == g(2),:);
    test_2(:,1:end-2) = test_2(:,1:end-2) ./ repmat(max(abs(test_2(:,1:end-2)),[],2),[1,T(dico)]);
    
    aux = [];
    for n = 1:n_test(g(1))
        aux = [aux; 0.5 * (repmat(test_1(n,1:end-2), [n_test(g(2)),1]) + test_2(:,1:end-2))];
    end
    test_dictionary(cnt+1:cnt+n_prod(k),:) = [aux, k * ones(n_prod(k),1),1*ones(n_prod(k),1)];
    cnt = cnt + n_prod(k);
end
test_dictionary(:,end-1) = test_dictionary(:,end-1) + n_class;

test_data = [aux_test; test_dictionary];
clear aux_test
%% Test 
true_class = test_data(:,end-1);

prior_g = zeros(1,n_class + n_comb);
for k = 1:(n_class + n_comb)
    prior_g(k) = sum(test_data(:,end-1) == k);
end
my_class = 1:(n_class + n_comb);
param.N_spect = N_spect(dico);
param.gpuFlag = 0;
n_buff = 1;
param.n_buff = n_buff;

tic
L_bay = identification_general(test_data, aux_L, prior_g, my_class, param);
elapsed_time = toc;

idx_p = L_bay > 0;
L_bay = L_bay(idx_p);
true_class = true_class(idx_p);

confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class+n_comb]);
mean(diag(BAY_confusionmatrix))
elapsed_time / size(database_test,1)
