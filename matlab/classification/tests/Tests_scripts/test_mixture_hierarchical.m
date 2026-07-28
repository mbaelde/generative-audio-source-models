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
if ~exist('feature_training')
    feature_training = aux_L_training;
    feature_test = aux_L_test;
    clear aux_L_training aux_L_test
    feature_training(:,1:end-2) = exp(feature_training(:,1:end-2));
end
clear feature_test database_training
%% Identify using a reduced dictionary
N = size(feature_training,1);

%for n_buff = 6:20
n_buff = 10;
%for N_sounds = [2,3,4,5,10,20,50,100,200,300,400];
N_sounds = 50;
feature_training(:,1:end-2) = feature_training(:,1:end-2) ./ repmat(sum(feature_training(:,1:end-2),2),[1,N_spect(dico)]);
feature_norm = create_reduced_dictionary(feature_training, N_sounds, database_folder, fold, 'single');

aux_L = feature_norm;
aux_L(:,1:end-2) = log(aux_L(:,1:end-2));
% Test on individual sounds
true_class = database_test(:,end-1);

prior_g = zeros(1,n_class);
for k = 1:n_class
    prior_g(k) = sum(database_test(:,end-1) == k);
end
my_class = 1:n_class;
param.N_spect = N_spect(dico);
param.gpuFlag = 0;
param.n_buff = n_buff;
param.dict = 0;
%%
tic
L_bay = identification_general(database_test, aux_L, prior_g, my_class, param);
elapsed_time = toc;

idx_p = L_bay > 0;
L_bay = L_bay(idx_p);
true_class = true_class(idx_p);

confusion_matrix = confusionmat(true_class, L_bay, 'order', 1:n_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class]);
mean(diag(BAY_confusionmatrix))
elapsed_time / size(database_test,1)
%%
save(['Results/Dictionary reduction/',database_folder,'Fold ',num2str(fold),'/result_nsound_',num2str(N_sounds),'_nbuff_',num2str(n_buff),'.mat'],'L_bay', 'BAY_confusionmatrix', 'elapsed_time')
%end
%end
%% Test on mixture
idx_class = cell(1,n_class);
dict_size = zeros(1,n_class);
for k = 1:n_class
    idx_class{k} = find(feature_training(:,end-1) == k);
    dict_size(k) = length(idx_class{k});
end
gm = [];
cnt = 1;
for k = 1:n_class
    for m = k+1:n_class
        gm(cnt,:) = [k,m];
        cnt = cnt + 1;
    end
end

prop_true = rand(2,1);
prop_true = prop_true ./ sum(prop_true);

nn = 1;
g = gm(nn,:);
idx_true = [idx_class{g(1)}(randi(dict_size(g(1)))), idx_class{g(2)}(randi(dict_size(g(2))))];
mixture_pdf = prop_true(1) * feature_training(idx_true(1),1:end-2) + prop_true(2) * feature_training(idx_true(2),1:end-2);

dict_size = size(aux_L,1);
aux_L_comp = aux_L(:,1:N_spect)';

n_class = length(my_class);
% get model size
model_size = zeros(1,n_class);
for k = 1:n_class
    model_size(k) = sum(aux_L(:,end-1) == k);
end
cum_model_size = cumsum(model_size);
msize = [0, cum_model_size];

prior_g = log(prior_g ./ sum(prior_g));

rspectrum_norm = repmat(mixture_pdf', [1,dict_size]);
L = sum(rspectrum_norm .* aux_L_comp);

for ii = 1:length(msize)-1
    A = L(msize(ii)+1:msize(ii+1));
    L_prior_max(ii) = max(A);
    if gpuFlag
        likelihood_group(ii) = gpuArray(log(sum(gather(exp(A - L_prior_max(ii))),'omitnan')));
    else
        likelihood_group(ii) = log(sum(exp(A - L_prior_max(ii)),'omitnan'));
    end
end
likelihood_group = likelihood_group + L_prior_max - log(model_size);

A = likelihood_group + prior_g;
L_prior_max = max(A);
norm_factor_g = L_prior_max + log(sum(exp(A - L_prior_max)));
posterior_g = -norm_factor_g + likelihood_group + prior_g;

[prob_sort,idx_sort] = sort(posterior_g);

disp('Predicted class:')
disp(num2str(fliplr(idx_sort(end-1:end))))
%%
tol = 1e-3;
iter_max = 300;
n_try = 1;
n_sounds = zeros(maxclust,2);
cnt = 1;
for k = 1:n_class
    for nn = 1:max(unique(idx_clusters{k}))
        n_sounds(cnt,:) = [sum(idx_clusters{k} == nn), k];
        cnt = cnt + 1;
    end
end
L_bay = zeros(1,size(database_test,1));
L_agg = zeros(1,size(database_test,1));

n_buff = 10;

true_class = database_test(:,end-1);
prop_class = zeros(n_class,size(database_test,1));

for b = 1:size(database_test,1)
    tic
    data = database_test(b,1:end-2);
    
    spectrum = abs(fft(data)).^2;
    spectrum_norm = N_spect * spectrum(1:N_spect) / sum(spectrum(1:N_spect));
    
    % Try which mixture has the bigger probability
    prop_em_mixt = em_algo_prop_mex(spectrum_norm, feature_norm, n_try, iter_max, tol, 0);
    
    prob_clusters = prop_em_mixt ./ n_sounds(:,1);
    prob_clusters = prob_clusters ./ sum(prob_clusters);
    
    for k = 1:n_class
        idx = find(n_sounds(:,2) == k);
        prop_class(k,b) = sum(prob_clusters(idx));
    end
    elapsed_time(b) = toc;
    L_bay(b) = find(prop_class(:,b) == max(prop_class(:,b) ));
    if mod(b,n_buff)==0
        sum_prop = sum(prop_class(:,b-n_buff+1:b),2);
        L_agg(b-n_buff+1:b) = find(sum_prop == max(sum_prop));
        figure(1)
        clf
        plot(true_class)
        hold on
        plot(L_agg)
        pause(0.001)
    end
    clc
    disp(['b: ',num2str(b),' / ',num2str(size(database_test,1))])
end

idx_p = L_bay > 0;
L_bay = L_bay(idx_p);
true_class = true_class(idx_p);

confusion_matrix = confusionmat(true_class, L_bay, 'order', 1:n_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class])
%%
use_Z = false;
method = 'ward';
load(['Results/Tree/',database_folder,'Fold ',num2str(fold),'/Z_',method,'.mat'])
Z = Z(:,1:3);

maxclust = round(N / N_sounds);
idx_clusters = cluster(Z, 'maxclust', maxclust);
%idx_clusters = reshape(repmat(1:maxclust,[N_sounds,1]),[1,maxclust*N_sounds]);
if length(idx_clusters) > N
    idx_clusters = idx_clusters(1:N);
else
    idx_clusters = [idx_clusters, idx_clusters(end)*ones(1,N-length(idx_clusters))];
end
%idx_clusters = idx_clusters(randperm(N));
my_feature = zeros(maxclust,N_spect);
for nn = 1:maxclust
    my_feature(nn,:) = mean(feature_training(idx_clusters == nn,1:end-2));
end
%%
tol = 1e-2;
iter_max = 300;
n_repl = 100;

%for nn = 1:size(gm,1)
nn = 1;
for repl = 1:n_repl
    prop_true = rand(2,1);
    prop_true = prop_true ./ sum(prop_true);
    
    g = gm(nn,:);
    idx_true = [idx_class{g(1)}(randi(dict_size(g(1)))), idx_class{g(2)}(randi(dict_size(g(2))))];
    mixture_pdf = prop_true(1) * feature_training(idx_true(1),1:end-2) + prop_true(2) * feature_training(idx_true(2),1:end-2);
    
    % prob = identification_mixture(mixture_pdf', aux_L, ones(1,n_class)/n_class);
    % [prob_sort, idx_sort] = sort(prob);
    
    % Try which mixture has the bigger probability
    n_try = 1;
    %disp('Mixture...')
    tic
    prop_em_mixt = em_algo_prop_mex(mixture_pdf, feature_norm, n_try, iter_max, tol, 0);
    elapsed_mixt = toc;
    
    [prop_sort,idx_em_mixt]=sort(prop_em_mixt);
    % Aggregate throught the different group
    n_sounds = zeros(maxclust,2);
    cnt = 1;
    for k = 1:n_class
        for nn = 1:max(unique(idx_clusters{k}))
            n_sounds(cnt,:) = [sum(idx_clusters{k} == nn), k];
            cnt = cnt + 1;
        end
    end
    
    prob_clusters = zeros(maxclust,1);
    for nn = 1:maxclust
        prob_clusters(nn) = prop_em_mixt(nn) / n_sounds(nn,1);
    end
    prob_clusters = prob_clusters ./ sum(prob_clusters);
    
    prop_class = zeros(n_class,1);
    for k = 1:n_class
        idx = find(n_sounds(:,2) == k);
        prop_class(k) = sum(prob_clusters(idx));
    end
    %posterior = prop_class .* prior ./ sum(prop_class .* prior);
    [~,idx_class_sort] = sort(prop_class);
    %disp('Classes sorted:')
    %disp(num2str(flipud(idx_class_sort)))
    save(['Results/Mixture/Hierarchical/Maxclust ',num2str(maxclust),'/result_',num2str(g(1)),'_',num2str(g(2)),'_repl_',num2str(repl),'.mat'],'idx_class_sort','elapsed_mixt')
    % % Take the two probable mixtures
    % midx = find(idx_clusters == idx_em_mixt(end));
    % midx = [midx; find(idx_clusters == idx_em_mixt(end-1))];
    %
    % % Try within the mixture which models are the most probable
    % disp('Within mixture...')
    % feature_mixt = feature_training(midx,1:end-2) ./ N_spect;
    %
    % n_try = 1;
    % tic
    % prop_em = em_algo_prop_mex(mixture_pdf, feature_mixt, n_try, iter_max, tol, 0);
    % elapsed_within = toc;
    %
    % % Sort the values and get the most probable models to mix
    % [prop_em_sort,idx_em]=sort(prop_em);
    %
    % candidate_feature = flipud(feature_training(midx(idx_em),1:end-2));
    % candidate_class = flipud(feature_training(midx(idx_em),end-1));
    % n_candidates = size(candidate_feature,1);
    %
    % % try some combinations
    % disp('Test combinations...')
    % n_try = 10;
    % iter_max = 10;
    % prop_em_candidate = zeros(2,1);
    % comb_tested = zeros(2,1);
    % L = 0;
    % cnt = 1;
    % tic
    % for cnt_1 = 1:n_candidates
    %     for cnt_2 = cnt_1+1:n_candidates
    %         comb_tested(:,cnt) = [candidate_class(cnt_1); candidate_class(cnt_2)];
    %
    %         if any(sum(repmat(comb_tested(:,cnt),[1,cnt-1]) - comb_tested(:,1:cnt-1)) == 0)
    %             continue;
    %         end
    %
    %         candidate_models(1,:) = candidate_feature(cnt_1,:) / N_spect;
    %         candidate_models(2,:) = candidate_feature(cnt_2,:) / N_spect;
    %
    %         prop_em_candidate(:,cnt) = em_algo_prop_mex(mixture_pdf, candidate_models, n_try, iter_max, tol, 0);
    %
    %         L(cnt) = sum(mixture_pdf .* log(prop_em_candidate(1,cnt) * candidate_models(1,:) + prop_em_candidate(2,cnt) * candidate_models(2,:)));
    %         cnt = cnt + 1;
    %         %disp(['cnt_1: ',num2str(cnt_1), ' / cnt_2: ',num2str(cnt_2)])
    %     end
    % end
    % elapsed_test = toc;
    % best = find(L == max(L));
    %
    % disp('Classes : True / Predicted')
    % class_true = feature_training(idx_true,end-1);
    % class_predicted = comb_tested(:,best);
    % [class_true, class_predicted]
    % disp('Prop : True / Predicted')
    % prop_predicted = prop_em_candidate(:,best);
    % [prop_true, prop_predicted]
    %
    % total_time = elapsed_mixt+elapsed_test+elapsed_within;
    % disp(['Total time: ',num2str(total_time),'s'])
    % save(['Results/Mixture/Hierarchical/Fold ',num2str(fold),'/result_',num2str(g(1)),'_',num2str(g(2)),'_repl_',num2str(repl),'.mat'],'class_true', 'class_predicted','prop_true', 'prop_predicted','total_time')
end
%end