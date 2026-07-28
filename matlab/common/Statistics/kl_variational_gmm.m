function dist = kl_variational_gmm(p_1, mu_1, sigma_1, p_2, mu_2, sigma_2)

M = length(p_1);
N = length(p_2);

D = zeros(M,N);
for m = 1:M
    for n = 1:N
        D(m,n) = kl_gaussian(mu_1(m), sigma_1(m), mu_2(n), sigma_2(n));
    end
end

D_p = zeros(M,M);
for m = 1:M
    for n = 1:M
        D_p(m,n) = kl_gaussian(mu_1(m), sigma_1(m), mu_1(n), sigma_1(n));
    end
end

dist = 0;
for m = 1:M
    dist = dist + p_1(m) * log( sum(p_1 .* exp( -D_p(m,:) )) / sum(p_2 .* exp( -D(m,:) )) );
end