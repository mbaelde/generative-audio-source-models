function [posterior_g,computation_time] = identification_kernel(database_test, feature_training, prior_g, param)
% Set parameters
n_fft = param.n_fft;
gpuFlag = param.gpuFlag;
power = param.power;
threshold = param.threshold;

n_samples = size(database_test,1);
n_models = size(feature_training,1);
n_class = max(feature_training(:,end-1));

% get model size
model_size = zeros(1,n_class);
for k = 1:n_class
    model_size(k) = sum(feature_training(:,end-1) == k);
end
cum_model_size = cumsum(model_size);
msize = [0, cum_model_size];

prior_g = log(prior_g ./ sum(prior_g));

% Initialize variables
if gpuFlag
    feature_training = gpuArray(feature_training);
    likelihood_group = zeros(1,n_class,'gpuArray');
    posterior_g = zeros(n_samples, n_class,'gpuArray');
else
    likelihood_group = zeros(1,n_class);
    posterior_g = zeros(n_samples, n_class);
end

% Identification
computation_time = zeros(1,n_samples);
for b = 1:n_samples
    tic
    if param.dict
        spectrum = database_test(b,1:end-2);
    else
        data = database_test(b,1:end-2);
        spectrum = abs(fft(data)).^power;
    end
    if 20*log10(sum(abs(data).^2)) < threshold
        posterior_g(b,:) = NaN;
        continue;
    else
        % Compute normalized spectrum
        spectrum_norm = spectrum(1:n_fft) ./ sum(spectrum(1:n_fft));
        spectrum_norm = log(spectrum_norm);
        if gpuFlag
            spectrum_norm = gpuArray(spectrum_norm);
        end
        
        % Compute all the likelihoods
        rspectrum_norm = repmat(spectrum_norm, [n_models,1]);
        
        B = sum(gammaln(feature_training(:,1:end-2)+1),2) - gammaln(sum(feature_training(:,1:end-2)+1,2));
        L = sum(rspectrum_norm .* feature_training(:,1:end-2),2) + B;
        
        % Aggregate per classes
        for ii = 1:n_class
            A = L(msize(ii)+1:msize(ii+1))';
            likelihood_group(ii) = LSE(A(A~=Inf));
        end
        likelihood_group = likelihood_group - log(model_size);
        
        % Use Bayes' rule
        posterior_g(b,:) = LSE_normprob(likelihood_group + prior_g);
    end
    computation_time(b) = toc;
    progressbar(b,n_samples)
end
