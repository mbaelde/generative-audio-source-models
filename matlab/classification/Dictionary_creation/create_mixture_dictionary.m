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


%% Enumerate the possibilities
gm = repmat((1:9)',[1,2]);
cnt = 10;
for k = 1:n_class
    for m = k+1:n_class
        gm(cnt,:) = [k,m];
        cnt = cnt + 1;
    end
end

n_comb = size(gm,1);
%% Create complete mixture dictionary based on reduced dictionary
% [2,3,4,5,10,20,50,100,200,300,400]
%N_sounds = [44    23     6     7     8    30    23    34    46]; % 98%
%N_sounds = [55  22  22  20  33  55  55  33  55]; % 97%
N_sounds = [91, 69, 53, 70, 70, 85, 105, 62, 100]; % 95%

database_training(:,1:end-2) = database_training(:,1:end-2) ./ repmat(max(abs(database_training(:,1:end-2)),[],2),[1,T(dico)]);
database_reduced = create_reduced_dictionary(database_training, N_sounds, database_folder, fold, 'single');

n_models = zeros(1,n_class);
for k = 1:n_class
    n_models(k) = sum(database_reduced(:,end-1) == k);
end

n_prod = zeros(1,n_comb);
for k = 1:n_comb
    n_prod(k) = prod(n_models(gm(k,:)));
end
n_mixture = sum(n_prod);
cum_prod = [0,cumsum(n_prod)];
%%
mixture_dict = zeros(n_mixture,T(dico) + 2);
for k = 1:n_comb
    g = gm(k,:);
    feature_1 = database_reduced(database_reduced(:,end-1) == g(1),:);
    feature_2 = database_reduced(database_reduced(:,end-1) == g(2),:);
    
    aux = zeros(n_prod(k),T(dico));
    for n = 1:n_models(g(1))
        aux((n-1)*n_models(g(2))+1:n*n_models(g(2)),:) = 0.5 * (repmat(feature_1(n,1:end-2), [n_models(g(2)),1]) + feature_2(:,1:end-2));
    end
    clear feature_1 feature_2

    my_feature = [aux, k*ones(size(aux,1),1), ones(size(aux,1),1)];
    mixture_dict((cum_prod(k)+1):cum_prod(k+1),:) = my_feature;
    progressbar(k,n_comb)
end

dictionary = fft(mixture_dict(:,1:end-2),[],2);
dictionary = abs(dictionary).^2;
dictionary = N_spect(dico) * dictionary(:,1:N_spect(dico)) ./ repmat(sum(dictionary(:,1:N_spect(dico)),2),[1,N_spect(dico)]);
dictionary = [dictionary,mixture_dict(:,end-1:end)];
aux_L = dictionary;
aux_L(:,1:end-2) = log(aux_L(:,1:end-2) ./ N_spect(dico));

%% Test on test set
% Create test set
aux_test = database_test(sort(randperm(size(database_test,1),floor(size(database_test,1)/20))),:);

n_test = zeros(1,n_class);
for k = 1:n_class
    n_test(k) = sum(aux_test(:,end-1) == k);
end

n_prod = zeros(1,n_comb);
for k = 1:n_comb
    n_prod(k) = prod(n_test(gm(k,:)));
end
cum_prod = [0,cumsum(n_prod)];
n_test_mixt = sum(n_prod);

test_dictionary = zeros(n_test_mixt, size(aux_test,2));

for k = 1:n_comb
    g = gm(k,:);
    test_1 = aux_test(aux_test(:,end-1) == g(1),:);
    test_1(:,1:end-2) = test_1(:,1:end-2) ./ repmat(max(abs(test_1(:,1:end-2)),[],2),[1,T(dico)]);
    test_2 = aux_test(aux_test(:,end-1) == g(2),:);
    test_2(:,1:end-2) = test_2(:,1:end-2) ./ repmat(max(abs(test_2(:,1:end-2)),[],2),[1,T(dico)]);
    
    aux = zeros(n_prod(k),T(dico));
    for n = 1:n_test(g(1))
        aux((n-1)*n_test(g(2))+1:n*n_test(g(2)),:) = 0.5 * (repmat(test_1(n,1:end-2), [n_test(g(2)),1]) + test_2(:,1:end-2));
    end
    aux = [aux, k * ones(n_prod(k),1),1*ones(n_prod(k),1)];
    test_dictionary((cum_prod(k)+1):cum_prod(k+1),:) = aux;

    progressbar(k,n_comb)
end

%% Test with all combinaison
true_class = test_dictionary(:,end-1);

prior_g = zeros(1,n_comb);
for k = (1:n_comb)
    prior_g(k) = sum(test_dictionary(:,end-1) == k);
end
my_class = (1:n_comb);
param.N_spect = N_spect(dico);
param.gpuFlag = 1;
n_buff = 1;
param.n_buff = n_buff;
param.dict = 0;

tic
[L_bay,posterior_g] = identification_general(test_dictionary, aux_L, prior_g, my_class, param);
elapsed_time = toc;
N = size(test_dictionary,1);

%for n_buff = 1:20 
    n_buff = 10;
    disp(['n_buff=',num2str(n_buff)])
    for b = 1:N
        if mod(b, n_buff) == 0
            sum_g = sum(gather(posterior_g((b-n_buff+1):b,:)),1,'omitnan');
            L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
        end
    end
    idx_p = L_bay > 0;
    L_bay = L_bay(idx_p);

    true_class = true_class(idx_p);
    
    confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,(n_comb)]);%(n_class+n_comb)]);
    gcr(n_buff) = mean(diag(BAY_confusionmatrix))*100;
%%
    save(['Results/Mixture/Generate dict/',database_folder,'Reduced dict 95/result_dataset_',num2str(dico),'_fold_',num2str(fold),'_nbuff_',num2str(n_buff),'_fromtemp.mat'], 'L_bay', 'BAY_confusionmatrix')
%end

%% Test on learning set
true_class = dictionary(:,end-1);

prior_g = zeros(1,n_comb);
for k = 1:n_comb
    prior_g(k) = sum(dictionary(:,end-1) == k);
end
my_class = 1:n_comb;
param.N_spect = N_spect(dico);
param.gpuFlag = 1;
n_buff = 1;
param.n_buff = n_buff;
param.dict = 1;

tic
[L_bay,posterior_g] = identification_general(dictionary, aux_L, prior_g, my_class, param);
elapsed_time = toc;

N = size(test_dictionary,1);

%for n_buff = 1:20 
    n_buff = 10;
    disp(['n_buff=',num2str(n_buff)])
    for b = 1:N
        if mod(b, n_buff) == 0
            sum_g = sum(gather(posterior_g((b-n_buff+1):b,:)),1,'omitnan');
            L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
        end
    end
    idx_p = L_bay > 0;
    L_bay = L_bay(idx_p);

    true_class = true_class(idx_p);
    
    confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,(n_comb)]);%(n_class+n_comb)]);
    gcr(n_buff) = mean(diag(BAY_confusionmatrix))*100;

%%

idx_p = L_bay > 0;
L_bay = L_bay(idx_p);
true_class = true_class(idx_p);

confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class+n_comb]);
mean(diag(BAY_confusionmatrix))
elapsed_time / size(database_test,1)
