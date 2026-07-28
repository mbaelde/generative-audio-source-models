function out = mixture_betapdf(x, a, b, weight)

M = length(weight);
N = size(x,1);

out = zeros(N,1);
for m = 1:M
    out = out + weight(m) * betapdf(x, a(m), b(m));
end
