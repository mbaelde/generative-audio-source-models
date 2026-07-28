function [idx_train, idx_test] = split_dataset_folds(idx_classes, percent_training)

id_class = unique(idx_classes);
n_class = length(id_class);

n_fold = floor(1 / (1 - percent_training));

c = cell(1,n_class);
for k = 1:n_class    
    n_samp = sum(idx_classes == id_class(k));
    c{k} = cvpartition(n_samp,'k',n_fold);
end

idx_train = cell(1,n_fold);
idx_test = cell(1,n_fold);
for ff = 1:n_fold
    idx_train{ff} = cell(1,n_class);
    idx_test{ff} = cell(1,n_class);
    for k = 1:n_class
        idx_train{ff}{k} = training(c{k},ff);
        idx_test{ff}{k} = test(c{k},ff);
    end

end