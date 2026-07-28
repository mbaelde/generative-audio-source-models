function out = mvbetapdf(x, a, b)

L = size(a,2);

out = 1;
for l = 1:L
    out = out .* betapdf(x(:,l),a(l),b(l));
end