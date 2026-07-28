function output = mixture_mixture_randn(N, moy, sig, prop_norm, prop_mixture)
%% mixture_mixture_randn
%
% This function samples N points of a mixture of mixture normal pdf of mu and sigma,
% with proportions weight within each mixture and prop_mixture for the
% mixture of mixture, over points x.
%
% Author: Maxime Baelde
% A-Volute // 2016
N_mixture = length(prop_mixture);

output = zeros(1,N);
for i = 1:N
    u = rand(1);
    if u < prop_mixture(1)
        output(i) = mixture_randn(1,moy(1,:), sig(1,:), prop_norm(1,:));
    else
        for k = 2:N_mixture
            if sum(prop_mixture(1:(k-1))) < u && u < sum(prop_mixture(1:k))
                output(i) = mixture_randn(1, moy(k,:), sig(k,:), prop_norm(k,:));
            end
        end
    end
end