function [L_min, L_map] = identification_general_non_interval(database, feature, my_class, param)

N_spect = param.N_spect;
winsize = param.winsize;
fs = param.fs;
f = 0:fs/winsize:fs/2;
type = param.type;

if strcmp(type,'clavel')
    n_buff = floor((0.5 * fs) / winsize);
else
    n_buff = 1;
end

n_class = length(my_class);

model_size = zeros(1,n_class);
for n = 1:length(feature)
    cl = feature(n).class;
    model_size(cl) = model_size(cl) + 1;
end
cum_model_size = cumsum(model_size);
% 
% aux_feature = [];
% for ii = 1:length(feature)
%     for jj = 1:length(feature{ii})
%         aux_feature = [aux_feature, feature{ii}{jj}];
%         disp([num2str(ii),', ',num2str(jj)])
%     end
% end

N = size(database,1);
dict_size = max(cum_model_size);

prior = -log(dict_size)*ones(1,dict_size);
L_min = zeros(1,N);
L_map = zeros(1,N);
L = zeros(n_buff, dict_size);
posterior = zeros(n_buff, dict_size);

sound_old = 0;
cnt = 0;
% Identification
for b = 1:N
    cnt = cnt + 1;
    % Calcul du spectre du buffer
    data = database(b,1:end-2)';
    sound = database(b,end);
    if sound ~= sound_old
        prior = -log(dict_size)*ones(1,dict_size);
        sound_old = sound;
    end
    
    % Compute spectrum
    spectrum = abs(fft(data)).^2;
    spectrum_norm = N_spect * spectrum(1:N_spect) ./ sum(spectrum(1:N_spect));
    x_obs = spectrum2hist(spectrum_norm, f(1:N_spect));
    % Calcul de la vraisemblance
    cnt_2 = 1;
    for ii = 1:length(feature)
        L(cnt,cnt_2) = sum(log(mixture_normpdf(x_obs, feature(ii).spectrum_model.mu, sqrt(feature(ii).spectrum_model.sigma), feature(ii).spectrum_model.mixing_coeff)));
        cnt_2 = cnt_2 + 1;
    end
    
    % Calcul de la posterior
    A = L(cnt,:) + prior;
    L_prior_max = max(A);
    norm_factor = L_prior_max + log(sum(exp(A - L_prior_max)));
    posterior(cnt,:) = -norm_factor + L(cnt,:) + prior;
    
    % Cherche la classe correspondance
    if mod(b, n_buff) == 0
        sum_L = sum(L,1);
        sum_post = sum(posterior,1);
        cnt = 0;
        
        idx_min = find(min(abs(sum_L)) == abs(sum_L),1);
        L_min((b-n_buff+1):b) = my_class(find(sign(idx_min-cum_model_size) <= 0,1));
        
        idx_map = find(max(sum_post) == sum_post,1);
        L_map((b-n_buff+1):b) = my_class(find(sign(idx_map-cum_model_size) <= 0,1));
        
        %prior = -log(dict_size)*ones(1,dict_size);
    else
        prior = posterior(cnt,:);
    end
    
    progressbar(b,N)
end

