function [posterior_g,computation_time] = identification(database_test, feature_training, prior_g, param)
% Set parameters
n_fft = param.n_fft;
gpuFlag = param.gpuFlag;
power = param.power;
threshold = param.threshold;
q = param.q;

n_samples = size(database_test,1);
n_models = size(feature_training,1);
id_class = unique(feature_training(:,end-1));
n_class = length(id_class);

% get model size
model_size = zeros(1,n_class);
for k = 1:n_class
    model_size(k) = sum(feature_training(:,end-1) == id_class(k));
end
cum_model_size = cumsum(model_size);
msize = [0, cum_model_size];

prior_g = log(prior_g ./ sum(prior_g));

% Initialize variables
if gpuFlag
    log_feature_training = gpuArray(log(feature_training(:,1:n_fft)));
    likelihood_group = zeros(1,n_class,'gpuArray');
else
    log_feature_training = log(feature_training(:,1:n_fft));
    likelihood_group = zeros(1,n_class);
end
% likelihood_group = zeros(1,n_class);
posterior_g = zeros(n_samples, n_class);
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
        spectrum_norm = q*spectrum(1:n_fft) ./ sum(spectrum(1:n_fft));
        %spectrum_norm = floor(q*spectrum(1:n_fft) ./ sum(spectrum(1:n_fft)));
%         figure(1)
%         clf
%         plot(spectrum_norm)
        
        if gpuFlag
            spectrum_norm = gpuArray(spectrum_norm);
        end
        
        % Compute all the likelihoods
        rspectrum_norm = repmat(spectrum_norm, [n_models,1]);
        L = sum(rspectrum_norm .* log_feature_training,2);
%         for ii = 1:n_class
%             rspectrum_norm = repmat(spectrum_norm, [model_size(ii),1]);
%             L = sum(rspectrum_norm .* log_feature_training(msize(ii)+1:msize(ii+1),1:end-2),2);
%             L = sum(rspectrum_norm .* gpuArray(log(feature_training(msize(ii)+1:msize(ii+1),1:end-2))),2);
        
        % Aggregate per classes
        for ii = 1:n_class
            A = L(msize(ii)+1:msize(ii+1))';
%             A = L';
            likelihood_group(ii) = LSE(gather(A));
        end
        likelihood_group = likelihood_group - log(model_size);
        
        % Use Bayes' rule
        posterior_g(b,:) = LSE_normprob(gather(likelihood_group + prior_g));
%         figure(2)
%         clf
%         plot(posterior_g(b,:))
%         pause(0.1)
    end
   
    computation_time(b) = toc;
    %clc
    %disp(['b: ',num2str(b),' / ',num2str(n_samples)])
    %save('tmp_result_2.mat','posterior_g','computation_time')
    progressbar(b,n_samples)
end
