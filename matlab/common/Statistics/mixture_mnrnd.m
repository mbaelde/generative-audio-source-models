function output = mixture_mnrnd(N, n, p, prop)
%% mixture_mnrnd
%
% This function samples N points of a mixture of multinomial pdf of n and p,
% with proportions weight.
%
% Author: Maxime Baelde
% A-Volute // 2016
K = length(prop);
M = size(p,1);
output = zeros(N,M);
for i = 1:N
    u = rand(1);
    if u < prop(1)
        output(i,:) = mnrnd(n,p(:,1));
    else
        for k = 2:K
            if sum(prop(1:(k-1))) < u && u < sum(prop(1:k))
                output(i,:) = mnrnd(n,p(:,k));
            end
        end
    end
end