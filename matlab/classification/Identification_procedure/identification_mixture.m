function proba = identification_mixture(mixture, aux_L, prior_g)
% Set parameters
N_spect = length(mixture);

dict_size = size(aux_L,1);

% Precompute things
aux_L_comp = aux_L(:,1:N_spect)';
my_class = unique(aux_L(:,end-1));

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
likelihood_group = zeros(1,length(msize)-1);
L_prior_max = zeros(1,length(msize)-1);

rspectrum_norm = repmat(mixture, [1,dict_size]);
L = sum(rspectrum_norm .* aux_L_comp);

for ii = 1:length(msize)-1
    A = L(msize(ii)+1:msize(ii+1));
    L_prior_max(ii) = max(A);
    likelihood_group(ii) = log(sum(exp(A - L_prior_max(ii)),'omitnan'));
end
likelihood_group = likelihood_group + L_prior_max - log(model_size);

A = likelihood_group + prior_g;
L_prior_max = max(A);
norm_factor_g = L_prior_max + log(sum(exp(A - L_prior_max)));
posterior_g = -norm_factor_g + likelihood_group + prior_g;

% Cherche la classe correspondance
sum_g = sum(gather(posterior_g),1,'omitnan');

proba = sum_g;
