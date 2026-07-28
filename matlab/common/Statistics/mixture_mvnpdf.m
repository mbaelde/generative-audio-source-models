function out = mixture_mvnpdf(x, mu, sigma, weight)

M = length(weight);

out = 0;
for m = 1:M
    out = out + weight(m) .* mvnpdf(x, mu(m,:), sigma(:,:,m));
end

