function identification_reduced_dictionary(database_folder, dico, fold, param)
T = param.T;
N_spect = param.N_spect;
fs = param.fs;
type = param.type;
n_class = param.n_class;
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

for N_sounds = fliplr([2,3,4,5,10,20,50,100,200,300,400]);
    disp(['N_sounds: ',num2str(N_sounds),])
    feature_norm = create_reduced_dictionary(feature_training, N_sounds, database_folder, fold, type);

    aux_L = feature_norm;
    aux_L(:,1:end-2) = log(aux_L(:,1:end-2));
    % Test on individual sounds
    true_class = database_test(:,end-1);

    prior_g = zeros(1,n_class);
    for k = 1:n_class
        prior_g(k) = sum(database_test(:,end-1) == k);
    end
    my_class = 1:n_class;
    param_i.N_spect = N_spect(dico);
    param_i.gpuFlag = 0;
    param_i.n_buff = 1;
    %%
    tic
    [L_bay,posterior_g] = identification_general(database_test, feature_training, prior_g, my_class, param_i);
    elapsed_time(N_sounds) = toc;
    N = size(database_test,1);
    L_bay = zeros(1,N);
    for n_buff = 1:20 
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

        true_class = true_class(idx_p);
        confusion_matrix = confusionmat(true_class, L_bay, 'order', 1:n_class);
        BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class]);
        mean(diag(BAY_confusionmatrix))*100;
        save(['Results/Dictionary reduction/',database_folder,type,'/Fold ',num2str(fold),'/result_nsound_',num2str(N_sounds),'_nbuff_',num2str(n_buff),'.mat'],'L_bay', 'BAY_confusionmatrix', 'elapsed_time')
    end
end
