function [L_bay,posterior_g,computation_time] = identification_general(database, aux_L, prior_g, my_class, param)
% Set parameters
N_spect = param.N_spect;
%type = param.type;
gpuFlag = param.gpuFlag;
n_buff = param.n_buff;

N = size(database,1);
dict_size = size(aux_L,1);

database = database';

% Precompute things
if gpuFlag
    aux_L_comp = gpuArray(aux_L(:,1:N_spect)');
else
    aux_L_comp = aux_L(:,1:N_spect)';
end

n_class = length(my_class);
% get model size
model_size = zeros(1,n_class);
for k = 1:n_class
    model_size(k) = sum(aux_L(:,end-1) == k);
end
cum_model_size = cumsum(model_size);

msize = [0, cum_model_size];

prior_g = log(prior_g ./ sum(prior_g));

% Initialize variables
L_bay = zeros(1,N);
if gpuFlag
    L = zeros(1, dict_size,'gpuArray');
    likelihood_group = zeros(1,length(msize)-1,'gpuArray');
    L_prior_max = zeros(1,length(msize)-1,'gpuArray');
    posterior_g = zeros(N, n_class,'gpuArray');
else
    L = zeros(1, dict_size);
    likelihood_group = zeros(1,length(msize)-1);
    L_prior_max = zeros(1,length(msize)-1);
    posterior_g = zeros(N, n_class);
end

% Identification
computation_time = zeros(1,N);
for b = 1:N
    tic
    if param.dict
        spectrum = database(1:end-2,b);
    else
        % Calcul du spectre du buffer
        data = database(1:end-2,b);
        % Compute spectrum
        spectrum = abs(fft(data)).^2;
    end
%     if 20*log10(sum(abs(data).^2)) < -60
%         posterior_g(b,:) = NaN;
%         continue;
%     else
    spectrum_norm = N_spect * spectrum(1:N_spect) ./ sum(spectrum(1:N_spect));%N_spect * spectrum(1:N_spect) ./ sum(spectrum(1:N_spect));
    if gpuFlag
        spectrum_norm = gpuArray(spectrum_norm);
    end
    
    rspectrum_norm = repmat(spectrum_norm, [1,dict_size]);
    L = sum(rspectrum_norm .* aux_L_comp);
    
    for ii = 1:length(msize)-1
        A = L(msize(ii)+1:msize(ii+1));
        L_prior_max(ii) = max(A);
        if gpuFlag
            likelihood_group(ii) = gpuArray(log(sum(gather(exp(A - L_prior_max(ii))),'omitnan')));
        else
            likelihood_group(ii) = log(sum(exp(A - L_prior_max(ii)),'omitnan'));
        end
    end
    likelihood_group = likelihood_group + L_prior_max - log(model_size);
    
    A = likelihood_group + prior_g;
    L_prior_max = max(A);
    norm_factor_g = L_prior_max + log(sum(exp(A - L_prior_max)));
    posterior_g(b,:) = -norm_factor_g + likelihood_group + prior_g;
    %end
    computation_time(b) = toc;
   
    % Cherche la classe correspondance
%     if mod(b, n_buff) == 0
%         if gpuFlag
%             sum_g = sum(gather(posterior_g((b-n_buff+1):b,:)),1,'omitnan');
%         else
%             sum_g = sum(posterior_g((b-n_buff+1):b,:),1,'omitnan');
%         end
%         L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
%         clc
%         disp(['iter: ',num2str(b),' / ',num2str(N)])
progressbar(b,N)
%     end
%     
end
