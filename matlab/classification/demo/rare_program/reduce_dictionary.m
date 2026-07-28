function feature_reduced = reduce_dictionary(feature_training, Z, N_sounds)

n_class = max(feature_training(:,end-1));

if length(N_sounds) == 1
    N_sounds = N_sounds * ones(1,n_class);
end

n_fft = size(feature_training,2)-2;

n_cluster = zeros(1,n_class);
idx_clusters = cell(1,n_class);

for k = 1:n_class
    feature_class = feature_training(feature_training(:,end-1) == k,:);
    n_cluster(k) = floor(size(feature_class,1) / N_sounds(k));
    idx_clusters{k} = cluster(Z{k}, 'maxclust', n_cluster(k));
end

maxclust = sum(n_cluster);
my_feature = zeros(maxclust,n_fft+2);
cnt = 1;
for k = 1:n_class
    feature_class = feature_training(feature_training(:,end-1) == k,:);
    for nn = 1:n_cluster(k)
        my_feature(cnt,:) = [mean(feature_class(idx_clusters{k} == nn,1:end-2),1),k,1];
        cnt = cnt + 1;
    end
end

feature_reduced = my_feature(:,1:end-2) ./ repmat(sum(my_feature(:,1:end-2),2),[1,n_fft]);
feature_reduced = [feature_reduced, my_feature(:,end-1:end)];
