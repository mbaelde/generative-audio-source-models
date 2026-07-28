function likelihood = mixture_dirichlet_likelihood(x, a, prop)

M = length(prop);
N = size(x,1);

logC = zeros(N,M); % log C
hat_a = sum(a,2);

for m = 1:M
    logC(:,m) = log(prop(m)) + gammaln(hat_a(m)) - sum(gammaln(a(m,:))) + sum(repmat((a(m,:) - 1),[N,1]) .* log(x),2);
end

likelihood = sum(LSE(logC),'omitnan');
