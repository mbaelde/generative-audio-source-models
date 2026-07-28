function x = univariate_truncated_randn(mu, mu_m, mu_p, sigma)
%% univariate_truncated_randn
%
% This function samples 1 points of an univariate truncatednormal pdf of mu and sigma,
% with lower bound mu_m and upper bound mu_p.
%
% Author: Maxime Baelde
% A-Volute // 2016

% mu : mean of the gaussian
% mu_m : lower bound
% mu_p : upper bound
% sigma : variance of the gaussian

%if mu_m * mu_p < 0 % C. Robert method
    %disp('Robert method')
    z = (mu_p - mu_m) * rand(1) + mu_m;

    if (mu_m < mu) && (mu < mu_p)
        rho = exp(- z*z / 2);
    elseif mu_p < mu
        rho = exp( ((mu_p * mu_p) - (z * z)) / 2);
    elseif mu < mu_m
        rho = exp( ((mu_m * mu_m) - (z * z)) / 2);
    end

    u = rand(1);

    while u > rho
        z = (mu_p-mu_m) * rand(1) + mu_m;

        if (mu_m < mu) && (mu < mu_p)
            rho = exp(- z*z / 2);
        elseif mu_p < mu
            rho = exp( ((mu_p * mu_p) - (z * z)) / 2);
        elseif mu < mu_m
            rho = exp( ((mu_m * mu_m) - (z * z)) / 2);
        end

        u = rand(1);
    end

    x = z;%mu + sqrt(sigma) * z;
% else % Repeated normal method
%     disp('Repeated normal')
%     z = mu + sqrt(sigma) * randn(1);
%     while (z < mu_m) || (mu_p < z)
%         z = mu + sqrt(sigma) * randn(1);
%     end
%     x = z;
% end
    