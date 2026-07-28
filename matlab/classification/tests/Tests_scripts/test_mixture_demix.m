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
fold = 5;

type = 'test';

%% Load data
load(['Database/',database_folder,'FS',num2str(fs),'/T',num2str(T(dico)),'/Uniform/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'])
if strcmp(type,'training')
    clear database_training feature_test
elseif strcmp(type,'test')
    clear database_training
end

N_spect = N_spect(dico);
%% Create reduced dictionary
N = size(feature_training,1);

n_buff = 10;
N_sounds = 50;

feature_norm = create_reduced_dictionary(feature_training, N_sounds, database_folder, fold, 'single');

aux_L = feature_norm;
aux_L(:,1:end-2) = log(aux_L(:,1:end-2));
%% Design a mixture
idx_class = cell(1,n_class);
dict_size = zeros(1,n_class);
for k = 1:n_class
    if strcmp(type,'training')
        idx_class{k} = find(feature_training(:,end-1) == k);
    elseif strcmp(type,'test')
        idx_class{k} = find(feature_test(:,end-1) == k);
    end
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

%%
aux_L = feature_training;
aux_L(:,1:end-2) = log(aux_L(:,1:end-2) / N_spect);

gpuFlag = 0;
n_buff = 10;
my_class = 1:n_class;
for k = 1:n_class
    prior_g(k) = sum(aux_L(:,end-1) == k);
end

red_dict_size = size(aux_L,1);
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

% Initialize variables
L = zeros(1,red_dict_size,'gpuArray');
L_bay = zeros(1,N);
likelihood_group = zeros(1,length(msize)-1);
L_prior_max = zeros(1,length(msize)-1);

%%
n_repl = 1000;
prop_m = [0.9,0.1;
        0.8,0.2;
        0.7,0.3;
        0.6,0.4;
        0.5,0.5];
for pp = 1:5
    prop = prop_m(pp,:);
    for nn = 1:size(gm,1)
        g = gm(nn,:);
        for repl = 1:n_repl
            idx_true = [idx_class{g(1)}(randi(dict_size(g(1)))), idx_class{g(2)}(randi(dict_size(g(2))))];
            if strcmp(type,'training')
                mixture_pdf = prop(1) * feature_training(idx_true(1),1:end-2) + prop(2) * feature_training(idx_true(2),1:end-2);
            elseif strcmp(type,'test')
                mixture_pdf = prop(1) * feature_test(idx_true(1),1:end-2) + prop(2) * feature_test(idx_true(2),1:end-2);
            end

            tic
            % Detect first class
            spectrum_norm = gpuArray(mixture_pdf');

            rspectrum_norm = repmat(spectrum_norm, [1,red_dict_size]);
            L = sum(rspectrum_norm .* aux_L_comp);
            [~,idx_L] = max(L);

            detected_class(1) = aux_L(idx_L,end-1);

            % Detect second class
            spectrum_norm = gpuArray(max(spectrum_norm - N_spect*exp(aux_L_comp(:,idx_L)),0));

            rspectrum_norm = repmat(spectrum_norm, [1,red_dict_size]);
            L = sum(rspectrum_norm .* aux_L_comp);
            [~,idx_L] = max(L);

            detected_class(2) = aux_L(idx_L,end-1);
            elapsed_time(nn,repl) = toc;
    %         disp('True / Predicted')
    %         disp(num2str([g',detected_class']))
    %         disp(['time: ',num2str(elapsed_time(nn,repl))])
            gcr(nn,repl) = (sum(diff([g',detected_class'],[],2)) == 0);
            clc
            disp(['--- g: ',num2str(g),' ---'])
            disp(['repl: ',num2str(repl),' / ',num2str(n_repl)])
        end
    end

    gcr_mean = 100 * (sum(gcr,2)/n_repl);

    if strcmp(type,'training')
        save(['Results/Mixture/Demix/',database_folder,'Fold ',num2str(fold),'/2 classes/Training/result_',num2str(prop(1)),'_',num2str(prop(2)),'.mat'],'gcr_mean','elapsed_time')
    elseif strcmp(type,'test')
        save(['Results/Mixture/Demix/',database_folder,'Fold ',num2str(fold),'/2 classes/Test/result_',num2str(prop(1)),'_',num2str(prop(2)),'.mat'],'gcr_mean','elapsed_time')
    end
end