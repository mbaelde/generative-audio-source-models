function output = mixture_mvnrnd(N, moy, sig, prop)
%% mixture_randn
%
% This function samples N points of a mixture of normal pdf of mu and sigma,
% with proportions weight.
%
% Author: Maxime Baelde
% A-Volute // 2016
K = length(prop);
M = size(moy,2);
output = zeros(N,M);
for i = 1:N
    u = rand(1);
    if u < prop(1)
        output(i,:) = mvnrnd(moy(1,:),sig(:,:,1));
    else
        for k = 2:K
            if sum(prop(1:(k-1))) < u && u < sum(prop(1:k))
                output(i,:) = mvnrnd(moy(k,:),sig(:,:,k));
            end
        end
    end
end