function out = mixture_mvncdf(xl,xu, mu, sigma, weight)

M = length(weight);

out = 0;
for m = 1:M
    out = out + weight(m) .* mvncdf(xl,xu, mu(m,:), sigma(:,:,m))';
end

