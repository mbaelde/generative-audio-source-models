function [cdf_test, model_test] = split_models(my_cdf, models, percent_training, N_models, model_size)

n_class = max(my_cdf(:,end));

cdf_test = [];
model_test = cell(1,round(N_models*(1-percent_training)));

cnt = 1;
count = 0;
for k = 1:n_class
    data_cdf = my_cdf(my_cdf(:,end) == k,:);
    idx_test = randperm(model_size(k), round(model_size(k)*(1-percent_training))); 
    cdf_test = [cdf_test; data_cdf(idx_test,:)];
    for m = 1:length(idx_test)
        model_test{cnt} = models{idx_test(m)+count};
        cnt = cnt + 1;
    end
    count = count + model_size(k);
    progressbar(k,n_class)
end