function r = mixture_drchrnd(prop, a, N)
%% mixture_drchrnd
%
% This function samples N points of a mixture of dirichlet pdf of parameters a,
% with proportions prop.
%
% Author: Maxime Baelde
% A-Volute // 2016
M = size(a,1);
K = size(a,2);

r = zeros(N,K);
for i = 1:N
    u = rand(1);
    if u < prop(1)
        r(i,:) = drchrnd(a(1,:),1);
    else
        for m = 2:M
            if sum(prop(1:(m-1))) < u && u < sum(prop(1:m))
                r(i,:) = drchrnd(a(m,:),1);
            end
        end
    end
end