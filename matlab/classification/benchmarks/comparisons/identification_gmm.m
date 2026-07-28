function [posterior_g,computation_time] = identification_gmm(database_test, model, descriptors, mean_training, std_training, coeff, param)

threshold = param.threshold;

n_class = max(database_test(:,end-1));
N = size(database_test,1);

prior_g = zeros(1,n_class);
for k = 1:n_class
    prior_g(k) = sum(database_test(:,end-1) == k);
end
prior_g = log( prior_g ./ sum(prior_g) );

L = zeros(1,n_class);
posterior_g = zeros(N, n_class);
computation_time = zeros(1,N);
for b = 3:N
    tic
    data = database_test(b-2:b,1:end-2);
    if mean(20*log10(sum(abs(data).^2,2))) < threshold
        posterior_g(b-2:b,:) = NaN;
        continue;
    else
        [feature_test, ~, ~, ~] = compute_descriptors(data, descriptors, param);
        
        if ~isempty(coeff)
            feature_test = feature_test * coeff;
        end
        
        for k = 1:n_class
            feature_test_norm = (feature_test - repmat(mean_training(k,:), [size(feature_test,1),1])) ./ repmat(std_training(k,:), [size(feature_test,1),1]);
            L(k) = LSE(mixture_mvnpdf(feature_test, model{k}.mu, model{k}.Sigma, model{k}.ComponentProportion)');
        end
        
        posterior_g(b,:) = LSE_normprob(L + prior_g);
    end
    computation_time(b) = toc;
    progressbar(b,N)
end
