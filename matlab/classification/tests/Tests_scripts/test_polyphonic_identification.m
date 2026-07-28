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
addpath('Polyphonic sounds')
%distcomp.feature( 'LocalUseMpiexec', false )
%gcp
%% Initialisation
dico = 3;
fold = 1;

%% Load data
load(['Database/',folder_database,'FS',num2str(fs(dico)),'/T',num2str(T(dico)),'/Metaclasse/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'],'raw_spectrum_test','raw_spectrum_training','database_test')
N_fft = T(dico)/2+1;
feature_training = [abs(raw_spectrum_training(:,1:N_fft)).^2, raw_spectrum_training(:,end-1:end)];
feature_test = [abs(raw_spectrum_test(:,1:N_fft)).^2, raw_spectrum_test(:,end-1:end)];
clear raw_spectrum_training raw_feature_test
feature_training(:,1:end-2) = feature_training(:,1:end-2) ./ repmat(sum(feature_training(:,1:end-2),2),[1,N_fft]);
feature_test(:,1:end-2) = N_fft .* feature_test(:,1:end-2) ./ repmat(sum(feature_test(:,1:end-2),2),[1,N_fft]);

class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};
n_class = length(class);

%% Enumerate the possibilities
% Single class
gm = repmat((1:n_class)',[1,2]);
% Mixture of 2 classes
cnt = n_class+1;
for k = 1:n_class
    for m = k+1:n_class
        gm(cnt,:) = [k,m];
        cnt = cnt + 1;
    end
end
% % Mixture of 3 classes
% for k = 1:n_class
%     for m = k+1:n_class
%         for n = m+1:n_class
%             gm(cnt,:) = [k,m,n];
%             cnt = cnt + 1;
%         end
%     end
% end

n_comb = size(gm,1);
%% Create complete mixture dictionary based on reduced dictionary
% [2,3,4,5,10,20,50,100,200,300,400]
%N_sounds = [44    23     6     7     8    30    23    34    46]; % 98%
%N_sounds = [55  22  22  20  33  55  55  33  55]; % 97%
%N_sounds = [91, 69, 53, 70, 70, 85, 105, 62, 100]; % 95%

% Metaclass
%N_sounds = [160  103   78   99   46]; % 95%
N_sounds = [162   91   73   83   41]; % 95.1%
%N_sounds = [92  58  44  23  22]; % 96%

feature_norm = create_reduced_dictionary(feature_training, N_sounds, folder_database, fold, 'single', 'feature');

n_models = zeros(1,n_class);
for k = 1:n_class
    n_models(k) = sum(feature_norm(:,end-1) == k);
end

n_prod = zeros(1,n_comb);
for k = 1:n_comb
    n_prod(k) = prod(n_models(gm(k,:)));
%     if k <= n_class
%         n_prod(k) = n_models(gm(k,1));
%     elseif n_class < k && k <= n_class+nchoosek(n_class,2)
%         n_prod(k) = prod(n_models(gm(k,1:2)));
%     elseif n_class+nchoosek(n_class,2) < k && k <= n_class+nchoosek(n_class,2)+nchoosek(n_class,3)
%         n_prod(k) = prod(n_models(gm(k,:)));
%     end
end
n_mixture = sum(n_prod);
cum_prod = [0,cumsum(n_prod)];
%% Create mixture dictionary (learning)
idx_clusters = cell(1,n_comb);

n_cluster = min(n_prod);
N_sounds_mixture = ceil(n_prod ./ n_cluster);

mixture_reduced = zeros(n_comb*n_cluster,N_fft + 2);

for k = 1:n_comb
    g = gm(k,:);
    feature_1 = feature_norm(feature_norm(:,end-1) == g(1),:);
    feature_2 = feature_norm(feature_norm(:,end-1) == g(2),:);
    
    aux = zeros(n_prod(k),N_fft);
    for n = 1:n_models(g(1))
        prop = rand(n_models(g(2)),2);
        prop = prop ./ repmat(sum(prop,2),[1,2]);
        prop_1 = repmat(prop(:,1),[1,N_fft]);
        prop_2 = repmat(prop(:,2),[1,N_fft]);
        aux((n-1)*n_models(g(2))+1:n*n_models(g(2)),:) = prop_1 .* repmat(feature_1(n,1:end-2), [n_models(g(2)),1]) + prop_2 .* feature_2(:,1:end-2);
    end
    clear feature_1 feature_2 prop_1 prop_2 prop
    
    % Reduce mixture dictionary
    feature_class = aux;
    clear aux
    Y = pdist(sqrt(feature_class/2));
    method = 'ward';
    Z = linkage(Y,method);
    save(['Clusters/',folder_database,'Metaclasse/Fold ',num2str(fold),'/Mixtures random/Z_',num2str(g(1)),'_',num2str(g(2)),'.mat'],'Z')
 %   load(['Clusters/',folder_database,'Metaclasse/Fold ',num2str(fold),'/Mixtures/Z_',num2str(g(1)),'_',num2str(g(2)),'.mat'],'Z')
    idx_clusters{k} = cluster(Z, 'maxclust', n_cluster);
    
    my_feature = zeros(n_cluster,N_fft+2);
    for nn = 1:n_cluster
        my_feature(nn,:) = [mean(feature_class(idx_clusters{k} == nn,:),1),k,1];
    end
    
%     my_feature = [aux, k*ones(size(aux,1),1), ones(size(aux,1),1)];
%     if g(1) == g(2)
%         my_feature = [aux, ones(size(aux,1),1), ones(size(aux,1),1)];
%     else
%         my_feature = [aux, 2*ones(size(aux,1),1), ones(size(aux,1),1)];
%     end
    my_feature(:,1:end-2) = my_feature(:,1:end-2) ./ repmat(sum(my_feature(:,1:end-2),2),[1,N_fft]);
    
    mixture_reduced((k-1)*n_cluster+1:k*n_cluster,:) = my_feature;
    progressbar(k,n_comb)
end

%mixture_reduced = create_reduced_dictionary(mixture_dictionary, N_sounds_mixture, database_folder, fold, 'mixture');
mixture_reduced(:,end-1) = mixture_reduced(:,end-1);
clear mixture_dictionary
% Complete dictionary
dictionary = mixture_reduced;
aux_L = dictionary;
aux_L(:,1:end-2) = log(aux_L(:,1:end-2));

%% Construct test_set
fid = fopen('Polyphonic sounds/labels.txt');
sound_list = cell(1);
start_time = [];
end_time = [];
label = cell(1);

tline = fgetl(fid);
cnt = 1;
while ischar(tline)
    content = strsplit(tline,'\t');
    sound_list{cnt} = content{1};
    start_time(cnt) = str2double(content{2});
    end_time(cnt) = str2double(content{3});
    label{cnt} = content{4};
    cnt = cnt + 1;
    tline = fgetl(fid);
end
fclose(fid);

sound_name = unique(sound_list);
n_sound = length(sound_name);

test_dataset = [];
for n = 1:n_sound
    % read sound file
    [audio,sr] = audioread(['Polyphonic sounds/',sound_name{n}]);
    % file where the name is in lists
    cnt = 1;
    idx_name = [];
    for nn = 1:length(sound_list)
        name = sound_list{nn};
        if strcmp(name, sound_name{n})
            idx_name = [idx_name,cnt];
        end
        cnt = cnt + 1;
    end
    % convert to mono
    if size(audio,2) >= 2
        audio = mean(audio,2);
    end
    hop_length = T(dico)/2;
    N = floor((length(audio)) ./ hop_length);
    % compute activation matrix based on onset and offset
    activation_matrix{n} = zeros(n_class,N);
    start_time_idx = round(start_time(idx_name) * sr / hop_length);
    start_time_idx(start_time_idx == 0) = 1;
    end_time_idx = round(end_time(idx_name) * sr / hop_length);
    end_time_idx(end_time_idx > N) = N;
    label_name = [];
    for nn = 1:length(idx_name);
        label_name{nn} = label{idx_name(nn)};
    end
    for nn = 1:length(label_name)
        idx_class = find(ismember(class, label_name{nn}));
        activation_matrix{n}(idx_class,start_time_idx(nn):end_time_idx(nn)) = 1;
    end
    %
    for b = 1:N
        active_class = find(activation_matrix{n}(:,b)' == 1);
        if isempty(active_class)
            idx_class = 0;
        elseif length(active_class) == 1
            idx_class = active_class;
        else
            idx_class = find(sum([gm(:,1) == active_class(1),(gm(:,2) == active_class(2))],2) == 2);
        end
        test_dataset = [test_dataset; audio((b-1)*D+1:(b-1)*D+T(dico))',idx_class,n];
    end
end

%% Test with all combinaison
true_class = test_dataset(:,end-1)';

%n_comb = 2;
prior_g = zeros(1,n_comb);
for k = (1:n_comb)
    prior_g(k) = sum(true_class == k);
end
my_class = (1:n_comb);
param.N_spect = N_spect(dico);
param.gpuFlag = 0;
n_buff = 1;
param.n_buff = n_buff;
param.dict = 0;

[L_bay,posterior_g,computation_time] = identification_general(test_dataset, aux_L, prior_g, my_class, param);
%%
N = size(test_dataset,1);

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
%true_class = test_data(:,end-1)';
true_class = true_class(idx_p);

N = length(true_class);
confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,(n_comb)]);%(n_class+n_comb)]);
gcr(n_buff) = mean(diag(BAY_confusionmatrix))*100;

matrix_true = zeros(n_class, N);
matrix_pred = zeros(n_class, N);
for n = 1:N
    matrix_true(unique(gm(true_class(n),:)),n) = 1;
    matrix_pred(unique(gm(L_bay(n),:)),n) = 1;
end

figure(1)
clf
imagesc(matrix_true)
figure(2)
clf
imagesc(matrix_pred)

% Segment-based metrics
[f1_score, error_rate] = metrics_sed(matrix_true, matrix_pred)

%% Test on learning set
nsound = 100;
rdict = zeros(n_comb*nsound,N_fft + 2);
for k = 1:n_comb
    dict_class = dictionary(dictionary(:,end-1) == k,:);
    n_dict_class = size(dict_class,1);
    rdict((k-1)*nsound+1:k*nsound,:) = dict_class(randperm(n_dict_class,nsound),:);
end
%rdict = dictionary;
true_class = rdict(:,end-1);

prior_g = zeros(1,n_comb);
for k = 1:n_comb
    prior_g(k) = sum(rdict(:,end-1) == k);
end
my_class = 1:n_comb;
param.N_spect = N_spect(dico);
param.gpuFlag = 1;
n_buff = 1;
param.n_buff = n_buff;
param.dict = 1;
N = size(rdict,1);
tic
[L_bay,posterior_g] = identification_general(rdict, aux_L, prior_g, my_class, param);
elapsed_time = toc;

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
%true_class = test_data(:,end-1)';
true_class = true_class(idx_p);

confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,(n_comb)]);%(n_class+n_comb)]);
gcr(n_buff) = mean(diag(BAY_confusionmatrix))*100;
gcr(n_buff)
%% Create mixture test
%aux_test = feature_test(sort(randperm(size(feature_test,1),floor(size(feature_test,1)/20))),:);
nsound = 30;
aux_test = zeros(n_class*nsound,N_fft + 2);
for k = 1:n_class
    dict_class = feature_test(feature_test(:,end-1) == k,:);
    n_dict_class = size(dict_class,1);
    aux_test((k-1)*nsound+1:k*nsound,:) = dict_class(randperm(n_dict_class,nsound),:);
end

n_test = zeros(1,n_class);
for k = 1:n_class
    n_test(k) = sum(aux_test(:,end-1) == k);
end
%n_comb = size(gm,1);
n_prod = zeros(1,n_comb);
for k = 1:n_comb
    n_prod(k) = prod(n_test(gm(k,:)));
end
cum_prod = [0,cumsum(n_prod)];
n_test_mixt = sum(n_prod);

mixture_test = zeros(n_test_mixt, size(aux_test,2));

for k = 1:n_comb
    g = gm(k,:);
    feature_1 = aux_test(aux_test(:,end-1) == g(1),:);
    feature_2 = aux_test(aux_test(:,end-1) == g(2),:);
    
    aux = zeros(n_prod(k),N_fft);
    for n = 1:n_test(g(1))
        aux((n-1)*n_test(g(2))+1:n*n_test(g(2)),:) = 0.5 * (repmat(feature_1(n,1:end-2), [n_test(g(2)),1]) + feature_2(:,1:end-2));
    end
    clear feature_1 feature_2
   
    my_feature = [aux, k*ones(size(aux,1),1), ones(size(aux,1),1)];
    my_feature(:,1:end-2) = N_spect(dico) * my_feature(:,1:end-2) ./ repmat(sum(my_feature(:,1:end-2),2),[1,N_fft]);
    
    mixture_test((cum_prod(k)+1):cum_prod(k+1),:) = my_feature;
    progressbar(k,n_comb)
end

% Complete dictionary
test_dictionary = mixture_test;

%% Test with all combinaison
true_class = test_dictionary(:,end-1);

%n_comb = 2;
prior_g = zeros(1,n_comb);
for k = (1:n_comb)
    prior_g(k) = sum(test_dictionary(:,end-1) == k);
end
my_class = (1:n_comb);
param.N_spect = N_fft;
param.gpuFlag = 0;
n_buff = 1;
param.n_buff = n_buff;
param.dict = 1;


[L_bay,posterior_g,computation_time] = identification_general(test_dictionary, aux_L, prior_g, my_class, param);

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
%true_class = test_data(:,end-1)';
true_class = true_class(idx_p);

confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,(n_comb)]);%(n_class+n_comb)]);
gcr(n_buff) = mean(diag(BAY_confusionmatrix))*100;
gcr(n_buff)

matrix_true = zeros(n_class, N);
matrix_pred = zeros(n_class, N);
for n = 1:N
    matrix_true(unique(gm(true_class(n),:)),n) = 1;
    matrix_pred(unique(gm(L_bay(n),:)),n) = 1;
end

% Segment-based metrics
[f1_score, error_rate] = metrics_sed(matrix_true, matrix_pred);

%%
save(['Results/Mixture/Generate dict/',database_folder,'Reduced dict 95/result_dataset_',num2str(dico),'_fold_',num2str(fold),'_nbuff_',num2str(n_buff),'_onspectrum.mat'], 'L_bay', 'BAY_confusionmatrix')
%end


%% Create test set
%aux_test = feature_test(sort(randperm(size(feature_test,1),floor(size(feature_test,1)/20))),:);
nsound = 50;
aux_test = zeros(n_class*nsound,T(dico) + 2);
for k = 1:n_class
    dict_class = database_test(database_test(:,end-1) == k,:);
    n_dict_class = size(dict_class,1);
    aux_test((k-1)*nsound+1:k*nsound,:) = dict_class(randperm(n_dict_class,nsound),:);
end
aux_test(:,1:end-2) = aux_test(:,1:end-2) ./ repmat(max(abs(aux_test(:,1:end-2)),[],2),[1,T(dico)]);

n_test = zeros(1,n_class);
for k = 1:n_class
    n_test(k) = sum(aux_test(:,end-1) == k);
end
%n_comb = size(gm,1);
n_prod = zeros(1,n_comb);
for k = 1:n_comb
    n_prod(k) = prod(n_test(gm(k,:)));
end
cum_prod = [0,cumsum(n_prod)];
n_test_mixt = sum(n_prod);

mixture_test = zeros(n_test_mixt, size(aux_test,2));

for k = 1:n_comb
    g = gm(k,:);
    feature_1 = aux_test(aux_test(:,end-1) == g(1),:);
    feature_2 = aux_test(aux_test(:,end-1) == g(2),:);
    
    aux = zeros(n_prod(k),T(dico));
    for n = 1:n_test(g(1))
        aux((n-1)*n_test(g(2))+1:n*n_test(g(2)),:) = 0.5 * (repmat(feature_1(n,1:end-2), [n_test(g(2)),1]) + feature_2(:,1:end-2));
    end
    clear feature_1 feature_2
   
    my_feature = [aux, k*ones(size(aux,1),1), ones(size(aux,1),1)];
    my_feature(:,1:end-2) = N_spect(dico) * my_feature(:,1:end-2) ./ repmat(sum(my_feature(:,1:end-2),2),[1,T(dico)]);
    
    mixture_test((cum_prod(k)+1):cum_prod(k+1),:) = my_feature;
    progressbar(k,n_comb)
end

% Complete dictionary
test_dataset = mixture_test;

%% Test with all combinaison
true_class = test_dataset(:,end-1);

%n_comb = 2;
prior_g = zeros(1,n_comb);
for k = (1:n_comb)
    prior_g(k) = sum(test_dataset(:,end-1) == k);
end
my_class = (1:n_comb);
param.N_spect = N_spect(dico);
param.gpuFlag = 0;
n_buff = 1;
param.n_buff = n_buff;
param.dict = 0;

[L_bay,posterior_g,computation_time] = identification_general(test_dataset, aux_L, prior_g, my_class, param);

N = size(test_dataset,1);

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
%true_class = test_data(:,end-1)';
true_class = true_class(idx_p);

confusion_matrix = confusionmat(true_class, L_bay, 'order', my_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,(n_comb)]);%(n_class+n_comb)]);
gcr(n_buff) = mean(diag(BAY_confusionmatrix))*100;

matrix_true = zeros(n_class, N);
matrix_pred = zeros(n_class, N);
for n = 1:N
    matrix_true(unique(gm(true_class(n),:)),n) = 1;
    matrix_pred(unique(gm(L_bay(n),:)),n) = 1;
end

figure(1)
clf
imagesc(matrix_true)
figure(2)
clf
imagesc(matrix_pred)

% Segment-based metrics
[f1_score, error_rate] = metrics_sed(matrix_true, matrix_pred)
