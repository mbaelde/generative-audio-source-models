function out = mixture_mvbetapdf(x, a, b, weight)

M = length(weight);

out = 0;
for m = 1:M
    out = out + weight(m) .* mvbetapdf(x, a(m,:), b(m,:));
end

