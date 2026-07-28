function feature_norm = create_reduced_dictionary(feature_training, N_sounds, database_folder, fold, type, data)

n_class = length(unique(feature_training(:,end-1)));

if length(N_sounds) == 1
    N_sounds = N_sounds * ones(1,n_class);
end

N_spect = size(feature_training,2)-2;

n_cluster = zeros(1,n_class);
idx_clusters = cell(1,n_class);

for k = 1:n_class
    feature_class = feature_training(feature_training(:,end-1) == k,:);
    n_cluster(k) = floor(size(feature_class,1) / N_sounds(k));
    % Clustering
    if strcmp(type,'single')
        if isempty(fold)
            %load(['Results/Dictionary reduction/',database_folder,'Clusters/Z_',num2str(k),'.mat'])
            load(['Results/Dictionary reduction/',database_folder,'Metaclasse/Z_',num2str(k),'.mat'])
        else
            %load(['Results/Mixture/Hierarchical/',database_folder,'Clusters/Fold ',num2str(fold),'/Z_',num2str(k),'.mat'])
            %load(['Results/Mixture/Hierarchical/',database_folder,'Metaclasse/Fold ',num2str(fold),'/Z_',num2str(k),'.mat'])
            load(['Clusters/',database_folder,'Metaclasse/Fold ',num2str(fold),'/Z_',num2str(k),'.mat'],'Z')
            %load(['Clusters/',database_folder,'Uniform/Fold ',num2str(fold),'/Z_',num2str(k),'.mat'],'Z')
        end
    elseif strcmp(type,'mixture')
        if isempty(fold)
            load(['Results/Dictionary reduction/',database_folder,'Mixtures/Z_',num2str(k),'.mat'])
        else
            load(['Results/Mixture/Hierarchical/',database_folder,'Mixtures/Fold ',num2str(fold),'/Z_',num2str(k),'.mat'])
        end
    end
    idx_clusters{k} = cluster(Z, 'maxclust', n_cluster(k));
    progressbar(k,n_class)
end

maxclust = sum(n_cluster);
my_feature = zeros(maxclust,N_spect+2);
cnt = 1;
for k = 1:n_class
    feature_class = feature_training(feature_training(:,end-1) == k,:);
    for nn = 1:n_cluster(k)
        my_feature(cnt,:) = [mean(feature_class(idx_clusters{k} == nn,1:end-2),1),k,1];
        cnt = cnt + 1;
    end
end

if strcmp(data,'raw')
    feature_norm = my_feature(:,1:end-2);
    feature_norm = [feature_norm, my_feature(:,end-1:end)];
elseif strcmp(data,'energy')
    feature_norm = my_feature(:,1:end-2);
elseif strcmp(data,'feature')
    feature_norm = my_feature(:,1:end-2) ./ repmat(sum(my_feature(:,1:end-2),2),[1,N_spect]);
    feature_norm = [feature_norm, my_feature(:,end-1:end)];
end