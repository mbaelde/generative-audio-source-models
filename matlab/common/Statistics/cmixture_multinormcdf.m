function out = cmixture_multinormcdf(x, mu, sigma, weight, M)

out = 0;
for m = 1:M
    out = out + weight(m) .* normcdf(x, mu(m,:), sigma(m,:));
end