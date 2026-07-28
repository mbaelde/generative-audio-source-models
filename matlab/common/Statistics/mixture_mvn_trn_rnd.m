function samples = mixture_mvn_trn_rnd(N, mu, mu_l, mu_u, sigma, prop)
%% mixture_mvn_trn_rdn
%
% This function samples N points of a mixture of normal pdf of mu and sigma,
% with proportions weight.
%
% Author: Maxime Baelde
% A-Volute // 2016
K = length(prop);
samples = zeros(N,K);

parfor i = 1:N
    u = rand(1);
    if u < prop(1)
        samples(i,:) = mvn_trn_rnd(mu(1,:), mu_l, mu_u, sigma(:,:,1));
    else
        for k = 2:K
            if sum(prop(1:(k-1))) < u && u < sum(prop(1:k))
                samples(i,:) = mvn_trn_rnd(mu(k,:), mu_l, mu_u, sigma(:,:,k));
            end
        end
    end
end