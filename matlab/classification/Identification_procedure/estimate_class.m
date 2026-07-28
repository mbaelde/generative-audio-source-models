function [L_min, L_map, prior] = estimate_class(b, data, winsize, N_spect, aux_L, prior, L_min, L_map, model_size, class)

% Compute spectrum
spectrum = abs(fft(data)).^2;
spectrum_norm = N_spect * spectrum(1:N_spect) ./ sum(spectrum(1:N_spect));

% Calcul de la vraisemblance
rspectrum_norm = repmat(spectrum_norm, [1,size(aux_L,1)]);
L = sum(rspectrum_norm .* (aux_L(:,1:N_spect)'));

% Calcul de la posterior
A = L + prior;
L_prior_max = max(A);
norm_factor = L_prior_max + log(sum(exp(A - L_prior_max)));
posterior = -norm_factor + L + prior;

% Cherche la classe correspondance
cum_model_size = cumsum(model_size);
idx_min = find(min(abs(L)) == abs(L),1);
L_min(1+(b-1)*winsize:winsize*b) = class(find(sign(idx_min-cum_model_size) <= 0,1));

idx_map = find(max(posterior) == posterior,1);
L_map(1+(b-1)*winsize:winsize*b) = class(find(sign(idx_map-cum_model_size) <= 0,1));
prior = posterior;
