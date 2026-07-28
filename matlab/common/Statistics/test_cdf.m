x = -5:0.01:5;
mu = [-1,1];
sigma = [1,1];
weight = [0.5,0.5];

for i = 1:10000
out = cmixture_uninormcdf(x, mu, sigma, weight);

out = cmixture_uninormcdf_mex(x, mu, sigma, weight);
end