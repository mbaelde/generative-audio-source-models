function [L_min, L_map, L_bay] = identification_complex(database, feature, my_class, param)
% Set parameters
N_spect = param.N_spect;
winsize = param.winsize;
fs = param.fs;
type = param.type;
f = param.f;
model_size = param.model_size;

load('Results\Performance\Complex\err_T1024')

if strcmp(type,'clavel')
    n_buff = floor((0.5 * fs) / winsize);
else
    n_buff = 1;
end

id_class = database(:,end-1);

n_class = length(my_class);
% get model size
% model_size = zeros(1,n_class);
% for k = 1:n_class
%     model_size(k) = sum(id_class == k);
% end
cum_model_size = cumsum(model_size);

idx_sound = database(:,end);
prior_g = zeros(1,n_class);
cnt = 1;
for k = 1:n_class
    class_data = idx_sound(database(:,end-1) == k);
    prior_g(k) = length(class_data);
    var = find(diff([0,class_data', max(class_data)+1]));
    for ii = 1:length(var)-1
        idx_s(cnt,:) = [k, ii, var(ii), var(ii+1)-1];
        cnt = cnt + 1;
    end
end

idx_class = idx_s(:,1);
cnt = 1;
var = find(diff([0,idx_class', max(idx_class)+1]));
for ii = 1:length(var)-1
    idx_g(ii,:) = [ii, var(ii), var(ii+1)-1];
end
msize = [0, cum_model_size];

prior_g = log(prior_g ./ sum(prior_g));

N = size(database,1);
dict_size = length(feature);
% Initialize variables
%prior = -log(dict_size)*ones(1,dict_size);
%L_min = zeros(1,N);
%L_map = zeros(1,N);
L_bay = zeros(1,N);
L = zeros(n_buff, dict_size);
%posterior = zeros(n_buff, dict_size);
posterior_g = zeros(n_buff, n_class);
likelihood_group = zeros(1,length(msize)-1);

%sound_old = 0;
cnt = 0;
idx = setdiff(1:length(feature),err);
cnt_feat = 1;
for i = 1:length(feature)
    if any(i == idx)
        myfeature{cnt_feat} = feature{i};
        cnt_feat = cnt_feat + 1;
    end
end

% Identification
disp('- 14029:18018-')
tic
for b = 1:10%N
    cnt = cnt + 1;
    % Calcul du spectre du buffer
    data = database(b,1:end-2)';
%     sound = database(b,end);
%     if sound ~= sound_old
%         prior = -log(dict_size)*ones(1,dict_size);
%         sound_old = sound;
%     end
%     
    % Compute spectrum
    spectrum = fft(data);
    spectrum = spectrum(1:N_spect);
    spectrum_data = [real(spectrum), imag(spectrum)];
    spectrum_norm = N_spect * spectrum_data ./ repmat(sum(abs(spectrum_data).^2),[N_spect,1]);
    data_to_reco = [f', spectrum_norm];
    
    % Calcul de la vraisemblance
    %cnt_err = 1;
%     err = [];
    aux = zeros(length(feature),1);
    parfor ii = 1:length(myfeature)
%         try
            aux(ii) = sum(log(mixture_mvnpdf(data_to_reco, myfeature{ii}.mu, myfeature{ii}.Sigma, myfeature{ii}.ComponentProportion)));
%         catch
%             err = [err;ii];
%         end
    end
    L(cnt,:) = aux;
    
    % Calcul de la posterior
%     A = L(cnt,:) + prior;
%     L_prior_max = max(A);
%     norm_factor = L_prior_max + log(sum(exp(A - L_prior_max)));
%     posterior(cnt,:) = -norm_factor + L(cnt,:) + prior;
    
%     prior = posterior(cnt,:);
    
    % for method 2
    for ii = 1:length(msize)-1
        A = L(cnt,msize(ii)+1:msize(ii+1));
        L_prior_max = max(A);
        likelihood_group(ii) = L_prior_max + log(sum(exp(A - L_prior_max))) - log(model_size(ii));
    end
    
    A = likelihood_group + prior_g;
    L_prior_max = max(A);
    norm_factor_g = L_prior_max + log(sum(exp(A - L_prior_max),'omitnan'));
    posterior_g(cnt,:) = -norm_factor_g + likelihood_group + prior_g;
    
    
    % Cherche la classe correspondance
    if mod(b, n_buff) == 0
%         sum_L = sum(L,1);
%         sum_post = sum(posterior,1);
        sum_g = sum(posterior_g,1);
        cnt = 0;
        
%         % Method 1 : Plug-In hat_S => hat_G
%         idx_min = find(min(abs(sum_L)) == abs(sum_L),1);
%         L_min((b-n_buff+1):b) = my_class(find(sign(idx_min-cum_model_size) <= 0,1));
%         
%         idx_map = find(max(sum_post) == sum_post,1);
%         L_map((b-n_buff+1):b) = my_class(find(sign(idx_map-cum_model_size) <= 0,1));
        
        % Method 2 : Full Bayesian
        L_bay((b-n_buff+1):b) = my_class(max(sum_g) == sum_g);
    end
    b
%     if mod(b, 10)
%         save('tmp_result_4_dico2_clavel.mat','L_min', 'L_map', 'L_bay')
%     end
    %progressbar(b,N)
end
toc