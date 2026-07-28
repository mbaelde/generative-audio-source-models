clear
clc

data_folder = '../../Data/';
database_folder = 'A-Volute/';

startup
addpath(genpath('Database'))
addpath(genpath('Statistics'))
addpath(genpath('Tree functions'))

%% Initialisation
dico = 3;
fold = 1;
%% Set the parameters
K = 2;                  % Number of classes
prop = ones(K,1) / K;   % Mixing proportions
G = [4, 9];             % Labels: Gunshot and Voice (male)

load(['Database/',database_folder,'T',num2str(T(dico)),'/Uniform/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'])
clear database_training feature_test
load('Results/Tree/A-Volute/Fold 1/Trees/tree_70440.mat')
labels = create_label_tree(subtree);

for k = 1:K
    idx_class{k} = find(database_test(:,end-1) == G(k));
end

database = database_test(union(idx_class{1}, idx_class{2}),:);
feature = feature_training(union(idx_class{1}, idx_class{2}),:);
clear database_test feature_training

for k = 1:K
    idx_class{k} = find(database(:,end-1) == G(k));
end
%% Choose randomly buffers
idx = [randi([min(idx_class{1}),max(idx_class{1})]), randi([min(idx_class{2}),max(idx_class{2})])];
% Create the mixtures
sounds = [database(idx(1),:);
          database(idx(2),:)];
      
mixture_sounds = sum(repmat(prop,[1,T(dico)]) .* sounds(:,1:end-2));
      
figure(1)
clf
plot(sounds(1,1:end-2))
hold on
plot(sounds(2,1:end-2))
plot(mixture_sounds)
legend(class{sounds(1,end-1)}, class{sounds(2,end-1)}, 'Mixture')

spectrums = [feature(idx(1),:);
             feature(idx(2),:)];
%spectrums = N_spect(dico) * spectrums ./ repmat(sum(spectrums,2),[1,N_spect(dico)]);

mixture_spectrums = sum(repmat(prop,[1,N_spect(dico)]) .* spectrums(:,1:end-2));
         
figure(2)
clf
plot(spectrums(1,1:end-2))
hold on
plot(spectrums(2,1:end-2))
plot(mixture_spectrums)
legend(class{sounds(1,end-1)}, class{sounds(2,end-1)}, 'Mixture')

%% Recover the classes
proba = [];
label_proba = [];

H = size(my_tree,1);

rspectrum = repmat(mixture_spectrums,[H,1]);
L = sum(rspectrum .* my_tree(:,1:end-2),2);
[~,idx_cluster] = max(L);
proba = [proba, L(idx_cluster)];
label_proba = [label_proba, {labels{1}{end}}];
if my_tree(idx_cluster,end-1) == 0
    L_bay = my_tree(idx_cluster,end);
else
    m = length(subtree{idx_cluster}) + 1;

    for ii = fliplr(1:m-1)
        if ii == m-1
            idchild = m-1;
        end
        L = sum(repmat(mixture_spectrums,[2,1]) .* subtree{idx_cluster}{idchild}(:,1:end-2),2);
        [~,idx] = max(L);
        proba = [proba, L(idx)];
        label_proba = [label_proba, {labels{idx_cluster}{idchild}}];
        if subtree{idx_cluster}{idchild}(idx,end-1) == 0
            L_bay = subtree{idx_cluster}{idchild}(idx,end);
            break;
        else
            idchild = subtree{idx_cluster}{idchild}(idx,end-1);
        end
    end
end

all_prob = proba ./ sum(proba);
[~, idx] = max(all_prob);
disp(['True: ',num2str(G)])
disp(['Predicted: '])
label_proba{idx}

%%
mixture_all = mean(feature_training(:,1:end-2));

