function likelihood = dirichlet_likelihood(r, a)

N = size(r,1);
hat_r = mean(log(r),1);
likelihood = N * (gammaln(sum(a)) - sum(gammaln(a)) + sum( (a-1) .* hat_r ));