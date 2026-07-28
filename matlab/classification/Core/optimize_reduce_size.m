clear
clc

data_folder = '../../Data/';
folder_database = 'A-Volute/';

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
class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};
n_class = length(class);
%% Load data
%load(['Database/',folder_database,'FS',num2str(fs),'/T',num2str(T(dico)),'/Uniform/80-20/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'],'database_test','feature_training')
%load(['Database/',folder_database,'FS',num2str(fs(dico)),'/T',num2str(T(dico)),'/Metaclasse/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'],'database_test','feature_training')
if strcmp(folder_database, 'A-Volute/')
    load(['Database/',folder_database,'FS',num2str(fs(dico)),'/T',num2str(T(dico)),'/Metaclasse/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'],'database_test','raw_spectrum_training')
else
    load(['Database/',folder_database,'FS',num2str(fs(dico)),'/T',num2str(T(dico)),'/Uniform/80-20/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'],'database_test','raw_spectrum_training')
end
N_fft = T(dico)/2+1;
feature_training = raw_spectrum_training(:,[1:N_fft, T(dico)+(1:2)]);
feature_training(:,1:end-2) = abs(feature_training(:,1:end-2)).^2 ./ N_fft;
%% Identify using a reduced dictionary
n_buff = 10;

prior_g = zeros(1,n_class);
my_class = 1:n_class;
param.N_spect = N_fft;
param.gpuFlag = 0;
param.n_buff = 1;
param.verbose = 1;
param.dict = 0;

reduce_size = [20,10,5,1];

th = 95.1;

for k = 1:n_class
    prior_g(k) = sum(database_test(:,end-1) == k);
end

N_sounds = [200,150,90,100,100];%300 * ones(1,n_class);
iter = 1;
stop = false;
while ~stop
    %feature_norm = create_reduced_dictionary(feature_training, N_sounds, folder_database, fold, 'single');
    feature_norm = create_reduced_dictionary(feature_training, N_sounds, folder_database, fold, 'single', 'feature');
    aux_L = feature_norm;
    aux_L(:,1:end-2) = log(aux_L(:,1:end-2));
    
    true_class = database_test(:,end-1);
    %tic
    [L_bay,posterior,computation_time] = identification_general(database_test, aux_L, prior_g, my_class, param);
    N = size(database_test,1);
    for b = 1:N
        if mod(b, n_buff) == 0
            sum_g = sum(gather(posterior((b-n_buff+1):b,:)),1,'omitnan');
            L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
        end
    end
    %toc
    
    idx_p = L_bay > 0;
    L_bay = L_bay(idx_p);
    true_class = true_class(idx_p);
    
    confusion_matrix = confusionmat(true_class, L_bay, 'order', 1:n_class);
    BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class]);
    gcr = diag(BAY_confusionmatrix)*100;
    for k = 1:n_class
        if gcr(k) < th
            if N_sounds(k) == 1        
                a_stop(k) = true;
            else
                a_stop(k) = false;
                if N_sounds(k) <= 20
                    N_sounds(k) = N_sounds(k) - 1;
                elseif iter < 5
                    N_sounds(k) = N_sounds(k) - reduce_size(1);
                elseif iter >= 5 && iter < 20
                    N_sounds(k) = N_sounds(k) - reduce_size(2);
                elseif iter >= 20 && iter < 50
                    N_sounds(k) = N_sounds(k) - reduce_size(3);
                else
                    N_sounds(k) = N_sounds(k) - reduce_size(4);
                end
            end
        elseif gcr(k) > th
            N_sounds(k) = N_sounds(k) + 1;
            a_stop(k) = true;
        else
            a_stop(k) = true;
        end
    end
    stop = all(a_stop) || iter > 100;
    iter = iter + 1;
    disp(['-----------'])
    disp(['iter: ',num2str(iter)])
    disp(['gcr: ',num2str(gcr')])
    disp(['N_sounds: ',num2str(N_sounds)])
    disp(['-----------'])
end
%N_sounds = [160  103   78   99   46];

%save(['Results/Dictionary reduction/',database_folder,'Fold ',num2str(fold),'/reduce_dict.mat'],'feature_norm','N_sounds')

%%
N_sounds = zeros(5,n_class);
for fold = 1:5
    file = load(['Results/Dictionary reduction/',database_folder,'Fold ',num2str(fold),'/reduce_dict.mat']);
    N_sounds(fold,:) = file.N_sounds;
end

opt_N_sounds = median(N_sounds);

load(['Features/',database_folder,'Uniform/T',num2str(T(dico)),'/feature_T',num2str(T(dico)),'.mat'])

feature_norm = create_reduced_dictionary(feature, opt_N_sounds, database_folder, 1, 'single');
aux_L = feature_norm;
aux_L(:,1:end-2) = log(aux_L(:,1:end-2));

% save to json
json = savejson('aux_L',aux_L);
f=fopen('aux_L.json','w');
fprintf(f,json);
fclose(f);

save('aux_L.mat','aux_L')