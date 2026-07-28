function Z = cluster_classes(feature)

n_class = max(feature(:,end-1));

Z = cell(1,n_class);
for k = 1:n_class
    feature_class = feature(feature(:,end-1) == k,:);
    % Clustering
    Y = pdist(sqrt(feature_class(:,1:end-2)/2));
    method = 'ward';
    Z{k} = linkage(Y,method);
end
