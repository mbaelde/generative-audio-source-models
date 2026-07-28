function L_bay = identification_general_gamma(database, aux_L, prior_g, my_class, param)
% Set parameters
N_spect = param.N_spect;
type = param.type;
gpuFlag = param.gpuFlag;
n_buff = param.n_buff;

% Precompute things
if gpuFlag
    aux_L_comp = gpuArray(aux_L(:,1:N_spect)');
else
    aux_L_comp = aux_L(:,1:N_spect)';
end

% if strcmp(type,'clavel')
%     n_buff = floor((0.5 * fs) / winsize);
% else
%     n_buff = 1;
% end

n_class = length(my_class);
% get model size
model_size = zeros(1,n_class);
for k = 1:n_class
    model_size(k) = sum(aux_L(:,end-1) == k);
end
cum_model_size = cumsum(model_size);

msize = [0, cum_model_size];

prior_g = log(prior_g ./ sum(prior_g));

N = size(database,1);
dict_size = size(aux_L,1);

% Initialize variables
L_bay = zeros(1,N);
if gpuFlag
    L = zeros(n_buff, dict_size,'gpuArray');
    likelihood_group = zeros(1,length(msize)-1,'gpuArray');
    L_prior_max = zeros(1,length(msize)-1,'gpuArray');
    posterior_g = zeros(n_buff, n_class,'gpuArray');
else
    L = zeros(n_buff, dict_size);
    likelihood_group = zeros(1,length(msize)-1);
    L_prior_max = zeros(1,length(msize)-1);
    posterior_g = zeros(n_buff, n_class);
end

database = database';

cnt = 0;
% Identification
for b = 1:N
    cnt = cnt + 1;
    % Calcul du spectre du buffer
    data = database(1:end-2,b);
    data = data ./ max(abs(data));
    
    % Compute spectrum
    spectrum = abs(fft(data)).^2;
    %spectrum_norm = N_spect * spectrum(1:N_spect) ./ sum(spectrum(1:N_spect));
    spectrum_norm = spectrum(1:N_spect);
    
    C_1 = gammaln(sum(spectrum_norm)+1) - sum(gammaln(spectrum_norm+1));  
    
    if gpuFlag
        spectrum_norm = gpuArray(spectrum_norm);
    end
    
    rspectrum_norm = repmat(spectrum_norm, [1,dict_size]);
     
    L(cnt,:) = sum(rspectrum_norm .* aux_L_comp) + C_1;
    
    for ii = 1:length(msize)-1
        A = L(cnt,msize(ii)+1:msize(ii+1));
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
    posterior_g(cnt,:) = -norm_factor_g + likelihood_group + prior_g;
    
    % Cherche la classe correspondance
    if mod(b, n_buff) == 0
        sum_g = sum(gather(posterior_g),1,'omitnan');
        cnt = 0;
        L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
%         figure(1)
%         clf
%         plot(database_test(:,end-1))
%         hold on
%         plot(L_bay)
%         pause(0.01)
    end

    progressbar(b,N)
end

