function r = mixture_mvbetarnd(prop, a, b, N)
%% mixture_drchrnd
%
% This function samples N points of a mixture of beta pdf of parameters a and b,
% with proportions prop.
%
% Author: Maxime Baelde
% A-Volute // 2016
M = size(a,1);
L = size(a,2);

r = zeros(N,L);
for i = 1:N
    u = rand(1);
    if u < prop(1)
        r(i,:) = mvbetarnd(a(1,:),b(1,:),1);
    else
        for m = 2:M
            if sum(prop(1:(m-1))) < u && u < sum(prop(1:m))
                r(i,:) = mvbetarnd(a(m,:),b(m,:),1);
            end
        end
    end
end