function output = mixture_randn(N, moy, sig, prop)
%% mixture_randn
%
% This function samples N points of a mixture of normal pdf of mu and sigma,
% with proportions weight.
%
% Author: Maxime Baelde
% A-Volute // 2016
K = length(prop);

output = zeros(1,N);
for i = 1:N
    u = rand(1);
    if u < prop(1)
        output(i) = moy(1) + sig(1)*randn(1);
    else
        for k = 2:K
            if sum(prop(1:(k-1))) < u && u < sum(prop(1:k))
                output(i) = moy(k) + sig(k)*randn(1);
            end
        end
    end
end