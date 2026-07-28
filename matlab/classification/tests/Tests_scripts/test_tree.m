clear
clc
database_folder = 'A-Volute/';
startup
addpath(genpath('Identification procedure'))
addpath(genpath('Database'))
addpath(genpath('Tree functions'))
% Data available:
% - class
% - D
% - T, N_fft, N_spect
% - fs
distcomp.feature( 'LocalUseMpiexec', false )
%%
dico = 3;

fold = 5;

disp('Prepare dictionary')
load(['Database/',database_folder,'FS',num2str(fs),'/T',num2str(T(dico)),'/Uniform/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'])
clear database_training feature_test

my_pdf = feature_training;
clear feature_training

%% Compute distances
m = size(my_pdf,1);

disp('Compute distances')
Y = pdist(sqrt(my_pdf(:,1:end-2)/2));

save(['Results/Tree/',database_folder,'Fold ',num2str(fold),'/distance.mat'],'Y','-v7.3')
%% Pair observations
disp('Pair observations')
method = 'ward';
Z = my_linkage(Y,method);
Z = [Z, m+(1:m-1)'];
save(['Results/Tree/',database_folder,'Fold ',num2str(fold),'/Z_',method,'.mat'],'Z','-v7.3')
%%
load(['Results/Tree/',database_folder,'Fold ',num2str(fold),'/Z_',method,'.mat'])
%% Complete Tree
disp('Process tree')
pdf_test = my_pdf(:,1:end-1);

type = 'uniform';
my_tree = create_tree(pdf_test, Z, type, m);

%% Choose height of the tree
pdf_test = my_pdf(:,1:end- 1);
pdf_test(:,1:end-1) = pdf_test(:,1:end-1) ./ repmat(sum(pdf_test(:,1:end-1),2),[1,N_spect(dico)]);
%[(0:10)+1,10*(2:10)+1,100*(2:10)+1,1000*(2:10)+1,20001,30001]
for iiii = [(0:10)+1,10*(2:10)+1,100*(2:10)+1,1000*(2:10)+1,20001,30001]
H = m-iiii;
disp([' ------ ',num2str(H), ' ----- '])
type = 'uniform';
[my_tree, subtree, Z_subtree] = create_tree_incomplete(Z, H, pdf_test, type);
%save(['Results/Tree/A-Volute/Fold ',num2str(fold),'/Trees/tree_',num2str(H),'.mat'],'my_tree','subtree','Z_subtree')

%% Tree based, majority vote
disp('Recognizing sounds')
param.N_spect = N_spect(dico);
param.n_buff = 1;

%[L_bay,elapsed_time] = identification_tree(database_test, my_tree, param);
elapsed_time = cell(1,2);
for test = 1:1
[L_bay,elapsed_time{test}] = identification_tree_incomplete(database_test, my_tree, subtree, param);
end
max_elapsed = max(cell2mat(elapsed_time'),[],2);
m_elapsed_time = elapsed_time{min(max_elapsed) == max_elapsed};
elapsed_time = m_elapsed_time;

idx_p = L_bay > 0;
L_bay = L_bay(idx_p);
true_class = database_test(:,end-1)';

true_class = true_class(idx_p);
confusion_matrix = confusionmat(true_class, L_bay, 'order', 1:n_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,n_class]);
mean(diag(BAY_confusionmatrix))*100
% figure(1)
% clf
% plot(elapsed_time)
[mean(elapsed_time), max(elapsed_time)]
%%
save(['Results/Tree/',database_folder,'Fold ',num2str(fold),'/Single/result_',method,'_sub',num2str(H),'.mat'],'L_bay','BAY_confusionmatrix', 'elapsed_time')
end

%% Create label tree
% labels = create_label_tree(subtree);
% labels = cell(1,length(subtree));
% 
% for ii = 1:length(subtree)
%     my_subtree = subtree{ii};
%     n_level = length(my_subtree);
%     
%     aux_labels = cell(1,n_level);
% 
%     for nn = 1:n_level
%         if my_subtree{nn}(1,end-1) == 0 && my_subtree{nn}(2,end-1) == 0
%             if my_subtree{nn}(1,end) == my_subtree{nn}(2,end)
%                 aux_labels{nn} = {num2str(my_subtree{nn}(1,end))};
%             else
%                 aux_labels{nn} = {num2str(my_subtree{nn}(1,end)), num2str(my_subtree{nn}(2,end))};
%             end
%         else
%             idx = find(my_subtree{nn}(:,end-1) == 0);
%             if idx == 1
%                 aux_labels{nn} = unique(sort([num2str(my_subtree{nn}(1,end)), aux_labels{my_subtree{nn}(2,end-1)}]));
%             elseif idx == 2
%                 aux_labels{nn} = unique(sort([aux_labels{my_subtree{nn}(1,end-1)}, num2str(my_subtree{nn}(2,end))]));
%             else
%                 aux_labels{nn} = unique(sort([aux_labels{my_subtree{nn}(1,end-1)}, aux_labels{my_subtree{nn}(2,end-1)}]));
%             end
%         end
%     end
%     labels{ii} = aux_labels;
%     progressbar(ii,length(subtree))
% end
%% Meta Tree
meta_tree = zeros(n_class,size(my_pdf,2)-1);
for k = 1:n_class
    data_class = my_pdf(my_pdf(:,end) == k,1:end-1);
    meta_tree(k,:) = log(mean(data_class));
end

%% Meta Tree based
%gcr = zeros(1,20);
%for n_buff = 1:20
n_buff = 1;
disp(['n_buff: ', num2str(n_buff)])
n_buffer = size(database,1);
L_bay = zeros(1,n_buffer);

my_class = 1:n_class;

prior_g = model_size;
prior_g = log(prior_g ./ sum(prior_g));

L = zeros(n_buff, n_class);
posterior_g = zeros(n_buff, n_class);

tic
cnt = 0;
for b = 1:n_buffer;
    cnt = cnt + 1;
    % Calcul du spectre du buffer
    data = database(b,1:end-2);
    % Compute spectrum
    spectrum = abs(fft(data)).^2;
    spectrum_norm = N_spect(dico) * spectrum(1:N_spect(dico)) ./ sum(spectrum(1:N_spect(dico)));
    rspectrum_norm = repmat(spectrum_norm,[n_class,1]);
    
    L(cnt,:) = sum(rspectrum_norm .* meta_tree,2);
    
    A = L(cnt,:) + prior_g;
    L_prior_max = max(A);
    norm_factor_g = L_prior_max + log(sum(exp(A - L_prior_max)));
    posterior_g(cnt,:) = -norm_factor_g + L(cnt,:) + prior_g;
    
    if mod(b, n_buff) == 0
        sum_g = sum(gather(posterior_g),'omitnan');
        cnt = 0;
        % Method 2 : Full Bayesian
        L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
    end
    
    progressbar(b,n_buffer)
end
elapsed_time = toc;

idx_zero = L_bay ~= 0;
L_bay = L_bay(idx_zero);
true_class = database(idx_zero,end-1);

aux_class = n_class;
confusion_matrix = confusionmat(true_class, L_bay, 'order', 1:aux_class);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,aux_class]);
mean(diag(BAY_confusionmatrix))*100
