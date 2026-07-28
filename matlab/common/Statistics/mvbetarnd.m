function sample = mvbetarnd(a,b,N)

L = length(a);

sample = zeros(N,L);
for n = 1:N
    for l = 1:L
        sample(n,l) = betarnd(a(l),b(l));
    end
end