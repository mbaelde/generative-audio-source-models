function x = mvn_trn_rnd(mu, mu_l, mu_u, sigma)
%% mvn_trn_rnd
%
% This function samples 1 points of an univariate truncatednormal pdf of mu and sigma,
% with lower bound mu_m and upper bound mu_p.
% (dimension 2 only)
%
% Author: Maxime Baelde
% A-Volute // 2016

% mu : mean of the gaussian
% mu_m : lower bound
% mu_p : upper bound
% sigma : variance of the gaussian

burn_in = 1000;

x = [0,0];

for i = 1:burn_in
    E_x = mu(1) + sigma(2,1)*(x(2)-mu(2))/sigma(2,2);
    E_y = mu(2) + sigma(1,2)*(x(1)-mu(1))/sigma(1,1);
    
    sigma_x = sigma(1,1) - sigma(2,1)^2/sigma(2,2);
    sigma_y = sigma(2,2) - sigma(1,2)^2/sigma(1,1);
    
    x(1) = univariate_truncated_randn(E_x, mu_l(1), mu_u(1), sigma_x);
    x(2) = univariate_truncated_randn(E_y, mu_l(2), mu_u(2), sigma_y);
end