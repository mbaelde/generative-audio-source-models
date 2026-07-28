function L_bay = identification_complex_discr(database, aux_L, my_class, param)
% Set parameters
N_spect = param.N_spect;
winsize = param.winsize;
fs = param.fs;
type = param.type;
f = param.f;

% Precompute things
aux_L_comp = aux_L';

if strcmp(type,'clavel')
    n_buff = floor((0.5 * fs) / winsize);
else
    n_buff = 1;
end

n_class = length(my_class);
% get model size
model_size = zeros(1,n_class);
for k = 1:n_class
    model_size(k) = sum(database(:,end-1) == k);
end
cum_model_size = cumsum(model_size);

idx_sound = database(:,end);
prior_g = zeros(1,n_class);
for k = 1:n_class
    class_data = idx_sound(database(:,end-1) == k);
    prior_g(k) = length(class_data);
end

msize = [0, cum_model_size];

prior_g = log(prior_g ./ sum(prior_g));

N = size(database,1);
dict_size = size(aux_L,1);
% Initialize variables
L_bay = zeros(1,N);
L = zeros(n_buff, dict_size);
posterior_g = zeros(n_buff, n_class);

cnt = 0;

load('Features\Clean database features\Complex\T1024\edges.mat')
n_x = length(edges{1})-1;
n_y = length(edges{2})-1;
n_z = length(edges{3})-1;
% Identification
for b = 1:N
    cnt = cnt + 1;
    % Calcul du spectre du buffer
    data = database(b,1:end-2);

    % Compute spectrum
    spectrum = fft(data);
    spectrum = spectrum(1:N_spect);
    spectrum_data = [real(spectrum)', imag(spectrum)'];
    spectrum_norm = N_spect * spectrum_data ./ repmat(sqrt(sum(abs(spectrum_data(:)).^2)),[N_spect,2]);
    data_to_model = [f', spectrum_norm];
    H = histcn(data_to_model,edges);
    if size(H,1) > n_x || size(H,2) > n_y || size(H,2) > n_z
        H = H(1:n_x,1:n_y,1:n_z);
    end
    H_u = sparse(repmat(H(:),[1,dict_size]));
    
    % Calcul de la vraisemblance
    L(cnt,:) = sum(H_u .* aux_L_comp,'omitnan');    
         
    % for method 2
    for ii = 1:length(msize)-1
        A = L(cnt,msize(ii)+1:msize(ii+1));
        L_prior_max = max(A);
        likelihood_group(ii) = L_prior_max + log(sum(exp(A - L_prior_max))) - log(model_size(ii));
    end
    
    A = likelihood_group + prior_g;
    L_prior_max = max(A);
    norm_factor_g = L_prior_max + log(sum(exp(A - L_prior_max)));
    posterior_g(cnt,:) = -norm_factor_g + likelihood_group + prior_g;
    
    % Cherche la classe correspondance
    if mod(b, n_buff) == 0
        sum_g = sum(posterior_g,1);
        cnt = 0;
              
        % Method 2 : Full Bayesian
        L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
    end
    
    if mod(b,10) == 1
        save('tmp_comp_discr.mat','L_bay')
    end
    
    progressbar(b,N)
end
