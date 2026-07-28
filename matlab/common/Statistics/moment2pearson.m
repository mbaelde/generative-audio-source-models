function [a, lambda, m, nu] = moment2pearson(mean_d, variance_d, skewness_d, kurtosis_d)

syms a lambda m nu r

eqn = [mean_d == lambda - ((a * nu) / r),...
       variance_d == (a^2 * (r^2 + nu^2)) / (r^2 * (r-1)),...
       skewness_d == ( (4*r) / (r-2) ) * sqrt( (r-1) / (r^2+nu^2) ),...
       kurtosis_d == ( 3 * (r-1) * ( (r+6) * (r^2+nu^2) - 8*r^2 ) ) / ( (r-2) * (r-3) * (r^2+nu^2) ),...
       r == 2*(m-1)];

[sola, sollambda, solm, solnu, solr] = solve(eqn);

a = eval(sola);
a = abs(a(1));
lambda = eval(sollambda);
lambda = lambda(find(lambda > 0,1));
m = eval(solm);
m = m(1);
nu = eval(solnu);
nu = abs(nu(1));