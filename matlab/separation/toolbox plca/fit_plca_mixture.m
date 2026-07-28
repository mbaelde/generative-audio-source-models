function mixture_act = fit_plca_mixture(V, z, dict, iter_max)

F = size(V,1);
mixture_act = rand(1,length(z));
mixture_act = mixture_act ./ sum(mixture_act);

for iter_cur = 1:iter_max
    % E-step
    posterior = repmat(mixture_act,[F,1]) .* dict.freq_basis;
    posterior = posterior ./ repmat(sum(posterior,2),[1,length(z)]);
    
    % M-step
    phi_t = sum(posterior .* repmat(V,[1,length(z)]),1);
    mixture_act = phi_t ./ repmat(sum(phi_t,2),[1,length(z)]);
end