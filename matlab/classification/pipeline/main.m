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
%dataset_name = 'TUT-SED-2017-DEV/';
dataset_name = 'BF/';
folder = [data_folder,dataset_name];

param.fs = 44100;
param.T = 2048;
param.D = 512;
param.n_fft = round(param.T / 5);
param.q = param.T / 2 + 1;
param.power = 2;
param.gpuFlag = 0;
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
        for k = 1:n_class
            feature_class = feature(feature(:,end-1) == id_class(k),:);
            database_class = database(database(:,end-1) == id_class(k),:);
            feature_training = [feature_training; feature_class(idx_train{fold}{k},:)];
            database_test = [database_test; database_class(idx_test{fold}{k},:)];
        end
        clear feature_class database_class
        
        %% Reduce the dictionary
        % if strcmp(status,'create')
        %     Z = cluster_classes(feature_training);
        %     save([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/Z_fold_',num2str(fold),'_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'Z')
        % elseif strcmp(status,'load')
        %     load([database_folder,dataset_name,'Folds/Set ',num2str(n_idx),'/Z_fold_',num2str(fold),'_fs',num2str(param.fs),'_T',num2str(param.T),'.mat'], 'Z')
        % end
        %
        % if param.use_reduced
        %     th.type = 'gcr';
        %     th.value = 96;
        %     th.initial_guess = [66   189   221    80   145];
        %     m = 10;
        %     N_sounds = optimize_reduce_size(feature_training, Z, database_test, m, th, param);
        %     % N_sounds = [29   152   163    53    87]; % 96%
        %     % N_sounds = [66   189   221    80   145];% 95%;
        %     % N_sounds = [83   227   238    97   162]; % 94.5%
        %     % N_sounds = [76   284   284   111   176]; %94%
        %     % N_sounds = [160  400   400   260   250];% 90%;
        %     %N_sounds = 2;
        %     feature_reduced = reduce_dictionary(feature_training, Z, N_sounds);
        % end
%         %%
%         feature_reduced = [];
%         n_clusters = 1;
%         cnt = 1;
%         for k  = 1:n_class
%             feature_class = feature_training(feature_training(:,end-1)==k,1:end-2);
%             n = size(feature_class,1);
%             steps = floor(n / n_clusters);
%             for nn = 1:n_clusters
%                 feature_reduced(cnt,:) = [mean(feature_class((nn-1)*steps+1:nn*steps,:)),k,1];
%                 cnt = cnt + 1;
%             end
%         end
        %% Identify new sounds
        id_class = setdiff(unique(database_test(:,end-1)),0);
        n_class = length(id_class);
        param.gpuFlag = 0;
        prior_g = zeros(1,n_class);
        param.use_reduced = 1;
        for k = 1:n_class
            prior_g(k) = sum(database_test(:,end-1) == id_class(k));
        end
        if param.use_reduced
            [posterior_g,computation_time] = identification(database_test, feature_reduced, prior_g, param);
        else
            [posterior_g,computation_time] = identification(database_test, feature_training, prior_g, param);
        end
        
m = 10;
N = size(posterior_g,1);
labels = zeros(1,N);
for b = 1:N
    if mod(b, m) == 0
        sum_g = sum(gather(posterior_g((b-m+1):b,:)),1,'omitnan');
        %sum_g = sum(gather(posterior_g((b-m+1):b,:)),1);
        %sum_g = exp(sum(log(posterior_g((b-m+1):b,:)),1,'omitnan'));
        
        if sum(sum_g) == 0
            labels((b-m+1):b) = 0;
%         elseif any(isnan(sum_g))
%             labels((b-m+1):b) = 0;
        else
            labels((b-m+1):b) = find(max(sum_g) == sum_g);
        end
    end
end

true_class = database_test(:,end-1)';
idx_zero = true_class == 0;
true_class = true_class(~idx_zero);
labels = labels(~idx_zero);
idx_zero = labels == 0;
true_class = true_class(~idx_zero);
labels = labels(~idx_zero);

N = length(labels);
%elapsed_time = elapsed_time(1:N);

matrix_true = zeros(3, N);
matrix_pred = zeros(3, N);
for n = 1:N
    if true_class(n) == 0
        matrix_true(:,n) = 0;
    else
        matrix_true(unique(gm(true_class(n),:)),n) = 1;
    end
%     if labels(n) == 0
%         matrix_pred(:,n) = 0;
%     else
        matrix_pred(unique(gm(labels(n),:)),n) = 1;
%     end
    
    progressbar(n,N)
end

figure(1)
clf
imagesc(matrix_true)
figure(2)
clf
imagesc(matrix_pred)
%
% Segment-based metrics
[f1_score, error_rate] = metrics_sed(matrix_true, matrix_pred)
for k = 1:n_class
    [f1_score_p(k), error_rate_p(k)] = metrics_sed(matrix_true(:,true_class == id_class(k)), matrix_pred(:,true_class == id_class(k)));
end
f1_score_p
%%
save(['Prototypes/Classification/Results/Full dictionary/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/result_nidx_',num2str(n_idx),'_fold_',num2str(fold),'.mat'],'f1_score','error_rate','posterior_g')
    end
end
        %% Assign labels
        m = 10;
        N = size(posterior_g,1);
        labels = zeros(1,N);
        for b = 1:N
            if mod(b, m) == 0
                sum_g = sum(gather(posterior_g((b-m+1):b,:)),1,'omitnan');
                %sum_g = exp(sum(log(posterior_g((b-m+1):b,:)),1,'omitnan'));
                if sum(sum_g) == 0
                    labels((b-m+1):b) = 0;
                else
                    labels((b-m+1):b) = find(max(sum_g) == sum_g);
                end
            end
        end
        %%
        m = 10;
        N = size(posterior_g,1);
        labels = cell(1,n_class);
        for k = 1:n_class
            id_samp = find(database_test(:,end-1) == k);
            N = length(id_samp);
            for b = 1:N
                if mod(b, m) == 0
                    sum_g = sum(gather(posterior_g(id_samp(b-m+1):id_samp(b),:)),1,'omitnan');
                    %sum_g = exp(sum(log(posterior_g((b-m+1):b,:)),1,'omitnan'));
                    if sum(sum_g) == 0
                        labels{k}((b-m+1):b) = 0;
                    else
                        labels{k}((b-m+1):b) = find(max(sum_g) == sum_g);
                    end
                end
            end
        end
        labels = cell2mat(labels);
        plot(labels)
        %%
        m = 10;
        N = size(posterior_g,1);
        labels = zeros(1,N);
        old_g = 0;
        alpha = 0.01;
        for b = 1:N
            if mod(b, m) == 0
                %sum_g = sum(gather(posterior_g((b-m+1):b,:)),1,'omitnan');
                aux_g = sum(gather(posterior_g(b-m+1:b,:)),1);
                sum_g = alpha * LSE_normprob(aux_g) + (1-alpha) * old_g;
                old_g = sum_g;
                %sum_g = exp(sum(log(posterior_g((b-m+1):b,:)),1,'omitnan'));
                if sum(sum_g) == 0
                    labels((b-m+1):b) = 0;
                else
                    labels((b-m+1):b) = find(max(sum_g) == sum_g);
                end
            end
        end
        %%
        old_g = 0;
        alpha = 0.1;
        for n = 1:size(posterior_g,1)
            if mod(n,m)==0
                aux_g = sum(gather(posterior_g(n-m+1:n,:)),1);
                sum_g = alpha * LSE_normprob(aux_g) + (1-alpha) * old_g;
                old_g = sum_g;
                figure(1)
                clf
                bar(sum_g)
                title(['True class: ',classes{true_class(n)},' (n: ',num2str(n),')'])
                pause(0.2)
            end
        end
        %% Compute metrics
        true_class = database_test(:,end-1)';
        %true_class = database_1(:,end-1)';
        idx_zero = true_class == 0;
        true_class = true_class(~idx_zero);
        labels = labels(~idx_zero);
        %true_class = database_mixture(idx_1_class,end-1)'+1;
        
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
        
        gcr = mean(diag(BAY_confusionmatrix))*100
        
        mean_time = median(computation_time)
        %%
        if param.use_reduced
            save(['Prototypes/Classification/Results/Reduced dictionary/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/result_fold_',num2str(fold),'_nidx_',num2str(n_idx),'_nsound_',num2str(N_sounds),'.mat'],'gcr','mean_time')
        else
            save(['Prototypes/Classification/Results/Full dictionary/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/result_fold_',num2str(fold),'_nidx_',num2str(n_idx),'.mat'],'gcr','mean_time')
            %save(['Prototypes/Classification/Results/Full dictionary/ESC-10 [New]/fs',num2str(param.fs),'_T',num2str(param.T),'/result_fold_',num2str(fold),'_nidx_',num2str(n_idx),'.mat'],'gcr','mean_time')
        end
    end
end
%%
aux_feature = feature;
feature_training = [];
cnt = 1;
%for k = [2,4,5]
id_class = setdiff(unique(aux_feature(:,end-1)),0);
for k = 1:length(id_class)
    feature_class = aux_feature(aux_feature(:,end-1) == id_class(k),:);
    feature_training = [feature_training; feature_class(:,1:end-2), id_class(k)*ones(size(feature_class,1),1), feature_class(:,end)];
    cnt = cnt + 1;
end

aux_feature = feature;
feature = [];
cnt = 1;
for k = [2,4,5]
    feature_class = aux_feature(aux_feature(:,end-1) == k,:);
    feature = [feature; feature_class(:,1:end-2), cnt*ones(size(feature_class,1),1), feature_class(:,end)];
    cnt = cnt + 1;
end


aux_feature = feature_reduced;
feature_reduced = [];
cnt = 1;
for k = [2,4,5]
    feature_class = aux_feature(aux_feature(:,end-1) == k,:);
    feature_reduced = [feature_reduced; feature_class(:,1:end-2), cnt*ones(size(feature_class,1),1), feature_class(:,end)];
    cnt = cnt + 1;
end

%% Polyphonic clasification
n_class = 3;
gm = repmat((1:n_class)',[1,3]);
% Mixture of 2 classes
cnt = n_class+1;
for k = 1:n_class
    for m = k+1:n_class
        gm(cnt,:) = [k,m,m];
        cnt = cnt + 1;
    end
end
% Mixture of 3 classes
for k = 1:n_class
    for m = k+1:n_class
        for n = m+1:n_class
            gm(cnt,:) = [k,m,n];
            cnt = cnt + 1;
        end
    end
end
n_comb = size(gm,1);

model_size = zeros(1,n_class);
for k = 1:n_class
    if param.use_reduced
        model_size(k) = sum(feature_reduced(:,end-1) == k);
    else
        model_size(k) = sum(feature_training(:,end-1) == k);
    end
end

feature_class = cell(1,n_class);
for k = 1:n_class
    if param.use_reduced
        feature_class{k} = feature_reduced(feature_reduced(:,end-1) == k,:);
    else
        feature_class{k} = feature_training(feature_training(:,end-1) == k,:);
    end
end

n_prod = zeros(1,n_comb);
n_comb_2 = nchoosek(n_class,2);
n_comb_3 = nchoosek(n_class,3);
for k = 1:n_comb
    if k <= n_class
        n_prod(k) = model_size(gm(k,1));
    elseif k > n_class && k <= n_class+n_comb_2
        n_prod(k) = prod(model_size(gm(k,1:2)));
    elseif k > n_class+n_comb_2
        n_prod(k) = prod(model_size(gm(k,:)));
    end
end
n_mixture = sum(n_prod);
cum_prod = [0,cumsum(n_prod)];

prior_g = zeros(1,n_class);
for k = 1:n_class
    prior_g(k) = sum(database_test(:,end-1) == k);
end
%% Create mixture dictionary using the reduced model
idx_clusters = cell(1,n_comb);

n_cluster = 200;%min(n_prod);
N_sounds_mixture = ceil(n_prod ./ n_cluster);

%mixture_reduced = feature_reduced;

n_comb_2 = nchoosek(n_class,2);
n_comb_3 = nchoosek(n_class,3);
mixture_2_class = zeros(n_comb_2*n_cluster,param.n_fft + 2);
Z = cell(1,n_comb_2);
for k = 1:n_comb_2
    g = gm(k+n_class,:);
    feature_1 = feature_reduced(feature_reduced(:,end-1) == g(1),:);
    feature_2 = feature_reduced(feature_reduced(:,end-1) == g(2),:);
    
    aux = zeros(n_prod(k+n_class),param.n_fft);
    for n = 1:model_size(g(1))
        aux((n-1)*model_size(g(2))+1:n*model_size(g(2)),:) = 0.5 .* (repmat(feature_1(n,1:end-2), [model_size(g(2)),1]) + feature_2(:,1:end-2));
    end
    clear feature_1 feature_2 prop_1 prop_2 prop
    
    % Reduce mixture dictionary
    feature_class = aux;
    clear aux
    Y = pdist(sqrt(feature_class/2));
    method = 'ward';
    Z{k} = linkage(Y,method);
    save(['Database/',dataset_name,'Mixture from reduced/Set ',num2str(n_idx),'/Z_2_fold_',num2str(fold),'.mat'],'Z')
    load(['Database/',dataset_name,'Mixture from reduced/Set ',num2str(n_idx),'/Z_2_fold_',num2str(fold),'.mat'],'Z')
    idx_clusters{k} = cluster(Z{k}, 'maxclust', n_cluster);
    
    my_feature = zeros(n_cluster,param.n_fft+2);
    for nn = 1:n_cluster
        my_feature(nn,:) = [mean(feature_class(idx_clusters{k} == nn,:),1),k+n_class,1];
    end
    
    my_feature(:,1:end-2) = my_feature(:,1:end-2) ./ repmat(sum(my_feature(:,1:end-2),2),[1,param.n_fft]);
    
    mixture_2_class((k-1)*n_cluster+1:k*n_cluster,:) = my_feature;
    progressbar(k,n_comb_2)
end

%mixture_reduced = [feature_reduced;mixture_2_class];

n_model_2 = zeros(1,n_comb_2);
for k = 1:n_comb_2
    n_model_2(k) = sum(mixture_2_class(:,end-1) == k+n_class);
end

% gm_3 = [6,3;
%         6,4;
%         6,5;
%         7,4;
%         7,5;
%         8,5;
%         10,4;
%         10,5;
%         11,5;
%         13,5];
gm_3 = [4,3];

n_cluster = 300;%min(n_prod);
mixture_3_class = zeros(n_comb_3*n_cluster,param.n_fft + 2);
Z = cell(1,n_comb_3);
for k = 1:n_comb_3
    g = gm_3(k,:);
    feature_1 = mixture_2_class(mixture_2_class(:,end-1) == g(1),:);
    feature_2 = feature_reduced(feature_reduced(:,end-1) == g(2),:);
    
    aux = zeros(size(feature_1,1) * size(feature_2,1),param.n_fft);
    for n = 1:size(feature_1,1)
        aux((n-1)*size(feature_2,1)+1:n*size(feature_2,1),:) = (2/3) .* (repmat(feature_1(n,1:end-2), [size(feature_2,1),1])) + (1/3) *(feature_2(:,1:end-2));
    end
    clear feature_1 feature_2 prop_1 prop_2 prop
    
    % Reduce mixture dictionary
    feature_class = aux;
    clear aux
    Y = pdist(sqrt(feature_class/2));
    method = 'ward';
    Z{k} = linkage(Y,method);
    save(['Database/',dataset_name,'Mixture from reduced/Set ',num2str(n_idx),'/Z_3_fold_',num2str(fold),'.mat'],'Z')
    load(['Database/',dataset_name,'Mixture from reduced/Set ',num2str(n_idx),'/Z_3_fold_',num2str(fold),'.mat'],'Z')
    idx_clusters{k} = cluster(Z{k}, 'maxclust', n_cluster);
    
    my_feature = zeros(n_cluster,param.n_fft+2);
    for nn = 1:n_cluster
        my_feature(nn,:) = [mean(feature_class(idx_clusters{k} == nn,:),1),k+n_class+n_comb_2,1];
    end
    
    my_feature(:,1:end-2) = my_feature(:,1:end-2) ./ repmat(sum(my_feature(:,1:end-2),2),[1,param.n_fft]);
    
    mixture_3_class((k-1)*n_cluster+1:k*n_cluster,:) = my_feature;
    progressbar(k,n_comb_3)
end

mixture_reduced = [feature_reduced; mixture_2_class; mixture_3_class];

aux_L = mixture_reduced;
aux_L(:,1:end-2) = log(aux_L(:,1:end-2));

n_models = size(aux_L,1);

model_size_r = zeros(1,n_class + n_comb_2 + n_comb_3);
for k = 1:n_class + n_comb_2 + n_comb_3
    model_size_r(k) = sum(aux_L(:,end-1) == k);
end
cum_model_size = cumsum(model_size_r);
msize = [0, cum_model_size];

%mixture_reduced = [feature_reduced; mixture_reduced(n_class*n_cluster+1:end,:)];
save('Database/A-Volute/mixture_reduced.mat','mixture_reduced')
%% Exhaustive method
idx_1 = sort(randperm(prior_g(true_mixt_label(1)), floor(prior_g(true_mixt_label(1))*0.1)));
idx_2 = sort(randperm(prior_g(true_mixt_label(2)), floor(prior_g(true_mixt_label(2))*0.01)));

cnt = 1;
est_class = [];
for n1 = idx_1
    for n2 = idx_2
        clc
        disp(['n1: ',num2str(find(n1==idx_1)), ' / ',num2str(length(idx_1))])
        disp(['n2: ',num2str(find(n2==idx_2)), ' / ',num2str(length(idx_2))])
        idx = [n1,n2];
        
        prop_true = [0.5,0.5];
        source = zeros(2,param.T);
        mixture_sound = zeros(1,param.T);
        for k = 1:2
            data_class = database_test(database_test(:,end-1) == true_mixt_label(k),:);
            source(k,:) = data_class(idx(k),1:end-2);
            source(k,:) = source(k,:) ./ max(abs(source(k,:)));
            mixture_sound = mixture_sound + prop_true(k) * source(k,:);
        end
        
        M = nchoosek(n_class,1) + nchoosek(n_class,2);
        prior_class = log(ones(1,M) ./  M);
        mixture_spectrum = abs(fft(mixture_sound)).^2;
        mixture_spectrum = param.n_fft * mixture_spectrum(1:param.n_fft) ./ sum(mixture_spectrum(1:param.n_fft));
        
        estimate_phiFlag = true;
        verbose = false;
        tic
        likelihood_group = compute_poly_label2(mixture_spectrum, feature_reduced, estimate_phiFlag, [0.5,0.5], verbose);
        toc
        posterior_group = LSE_normprob(likelihood_group + prior_class);
        est_class(cnt) = find(posterior_group == max(posterior_group));
        cnt = cnt + 1;
    end
end

%% Reduced using reduced
for gg = 6:15
    true_mixt_label = gm(gg,:);
    
    idx_1 = sort(randperm(prior_g(true_mixt_label(1)), floor(prior_g(true_mixt_label(1)))));%*0.1)));
    idx_2 = sort(randperm(prior_g(true_mixt_label(2)), floor(prior_g(true_mixt_label(2)))));%*0.01)));
    
    prop_true = [0.5,0.5];
    M = n_class + n_comb_2 + n_comb_3;
    prior_class = log(ones(1,M) ./  M);
    
    cnt = 1;
    est_class = zeros(1,length(idx_1)*length(idx_2));
    posterior_group = zeros(length(idx_1)*length(idx_2),M);
    elapsed_time = zeros(1,length(idx_1)*length(idx_2));
    for n1 = 986:idx_1(end)
        for n2 = 1:10%idx_2
            clc
            disp(['-- gg: ',num2str(gg),' --'])
            disp(['n1: ',num2str(find(n1==idx_1)), ' / ',num2str(length(idx_1))])
            disp(['n2: ',num2str(find(n2==idx_2)), ' / ',num2str(length(idx_2))])
            idx = [n1,n2];
            
            source = zeros(2,param.T);
            mixture_sound = zeros(1,param.T);
            for k = 1:2
                data_class = database_test(database_test(:,end-1) == true_mixt_label(k),:);
                source(k,:) = data_class(idx(k),1:end-2);
                source(k,:) = source(k,:) ./ max(abs(source(k,:)));
                mixture_sound = mixture_sound + prop_true(k) * source(k,:);
            end
            
            tic
            mixture_spectrum = abs(fft(mixture_sound)).^2;
            mixture_spectrum = param.n_fft * mixture_spectrum(1:param.n_fft) ./ sum(mixture_spectrum(1:param.n_fft));
            
            likelihood_group = zeros(1,M);
            rmixture_spectrum = repmat(mixture_spectrum, [sum(model_size_r),1]);
            L = sum(rmixture_spectrum .* aux_L(:,1:end-2),2);
            for k = 1:M
                A = L(msize(k)+1:msize(k+1))';
                likelihood_group(k) = LSE(A);
            end
            likelihood_group = likelihood_group - log(model_size_r);
            
            posterior_group(cnt,:) = LSE_normprob(likelihood_group + prior_class);
            if mod(cnt,10) == 0
                sum_posterior = sum(posterior_group(cnt-9:cnt,:),1);
                est_class(cnt-9:cnt) = find(sum_posterior == max(sum_posterior));
            end
            elapsed_time(cnt) = toc;
            cnt = cnt + 1;
            
        end
        save(['result_poly_gg_',num2str(gg),'.mat'],'est_class','elapsed_time')
    end
end

%% F1-score and error rate per class
true_class = 14;
load(['Prototypes/Classification/Results/Polyphonic/A-Volute/fs44100_T2048/result_poly_gg_',num2str(true_class),'.mat'])
est_class = est_class(est_class > 0);
N = length(est_class);
%elapsed_time = elapsed_time(1:N);
n_class = 5;
gm = repmat((1:n_class)',[1,2]);
% Mixture of 2 classes
cnt = n_class+1;
for k = 1:n_class
    for m = k+1:n_class
        gm(cnt,:) = [k,m];
        cnt = cnt + 1;
    end
end
n_comb = size(gm,1);

matrix_true = zeros(n_class, N);
matrix_pred = zeros(n_class, N);
for n = 1:N
    matrix_true(unique(gm(true_class,:)),n) = 1;
    matrix_pred(unique(gm(est_class(n),:)),n) = 1;
    progressbar(n,N)
end

figure(1)
clf
imagesc(matrix_true)
figure(2)
clf
imagesc(matrix_pred)

% Segment-based metrics
[f1_score, error_rate] = metrics_sed(matrix_true, matrix_pred)

% 1   : f = 0.9896, e = 0.0210
% 2   : f = 0.9210, e = 0.1490
% 3   : f = 0.9409, e = 0.0881
% 4   : f = 0.9329, e = 0.1066
% 5   : f = 0.9582, e = 0.0591
% 1+2 : f = 0.8763, e = 0.1268
% 1+3 : f = 0.7840, e = 0.2177
% 1+4 : f = 0.8517, e = 0.1505
% 1+5 : f = 0.8610, e = 0.1438
% 2+3 : f = 0.7658, e = 0.3374
% 2+4 : f = 0.7475, e = 0.3201
% 2+5 : f = 0.7272, e = 0.3115
% 3+4 : f = 0.7016, e = 0.3445
% 3+5 : f = 0.7044, e = 0.3400
% 4+5 : f = 0.8129, e = 0.2043

%% F1-score and error rate for all experiment
n_class = 5;
gm = repmat((1:n_class)',[1,2]);
% Mixture of 2 classes
cnt = n_class+1;
for k = 1:n_class
    for m = k+1:n_class
        gm(cnt,:) = [k,m];
        cnt = cnt + 1;
    end
end
n_comb = size(gm,1);

labels = [];
true_class = [];
for kk = 6:n_comb
    load(['Prototypes/Classification/Results/Polyphonic/A-Volute/fs44100_T2048/result_poly_gg_',num2str(kk),'.mat'])
    est_class = est_class(est_class > 0);
    labels = [labels, est_class];
    true_class = [true_class, kk * ones(1,length(est_class))];
end

%%
%for m = 1:50;
m = 10;
N = size(posterior_g,1);
labels = zeros(1,N);
for b = 1:N
    if mod(b, m) == 0
        sum_g = sum(gather(posterior_g((b-m+1):b,:)),1,'omitnan');
        %sum_g = exp(sum(log(posterior_g((b-m+1):b,:)),1,'omitnan'));
        if sum(sum_g) == 0
            labels((b-m+1):b) = 0;
        else
            labels((b-m+1):b) = find(max(sum_g) == sum_g);
        end
    end
end

true_class = database_test(:,end-1)';
idx_zero = true_class == 0;
true_class = true_class(~idx_zero);
labels = labels(~idx_zero);
idx_zero = labels == 0;
true_class = true_class(~idx_zero);
labels = labels(~idx_zero);

N = length(labels);
%elapsed_time = elapsed_time(1:N);

matrix_true = zeros(6, N);
matrix_pred = zeros(6, N);
for n = 1:N
    matrix_true(unique(gm(true_class(n),:)),n) = 1;
    matrix_pred(unique(gm(labels(n),:)),n) = 1;
    progressbar(n,N)
end

figure(1)
clf
imagesc(matrix_true)
figure(2)
clf
imagesc(matrix_pred)
%
% Segment-based metrics
[f1_score, error_rate] = metrics_sed(matrix_true, matrix_pred);
save(['Prototypes/Classification/Results/Full dictionary/',dataset_name,'fs',num2str(param.fs),'_T',num2str(param.T),'/result_nidx_',num2str(n_idx),'_fold_',num2str(fold),'.mat'],'f1_score','error_rate','posterior_g')

%end
% mono : f = 0.9390, e = 0.990
% mixture : f = 0.7449, e = 0.3059

%%
feature_1 = feature(feature(:,end-1) == 1,:);
feature_2 = feature(feature(:,end-1) == 2,:);
idx_1 = randi(size(feature_1,1));
idx_2 = randi(size(feature_2,1));
x_1 = feature_1(idx_1,1:end-2);
x_2 = feature_2(idx_2,1:end-2);
figure
plot(x_1)
hold on
plot(x_2)
phi = 0.4;
x = phi * x_1 + (1 - phi) * x_2;
plot(x)
q = 1e6;
x_q = q*x;

phi_est = 0.5;
log_tau(1) = LSE(log(phi_est) + sum(x_q.*log(x_1)));
log_tau(2) = LSE(log(1-phi_est) + sum(x_q.*log(x_2)));

phi_est = 