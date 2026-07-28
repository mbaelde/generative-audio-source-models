function r = mixture_betarnd(prop, a, b, N)
%% mixture_drchrnd
%
% This function samples N points of a mixture of beta pdf of parameters a and b,
% with proportions prop.
%
% Author: Maxime Baelde
% A-Volute // 2016
M = size(a,1);

r = zeros(N,1);
for i = 1:N
    u = rand(1);
    if u < prop(1)
        r(i) = betarnd(a(1),beta(1));
    else
        for m = 2:M
            if sum(prop(1:(m-1))) < u && u < sum(prop(1:m))
                r(i) = betarnd(a(m),b(m));
            end
        end
    end
end