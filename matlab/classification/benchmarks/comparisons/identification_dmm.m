function [posterior_g,computation_time] = identification_dmm(database_test, model, param)

power = param.power;
threshold = param.threshold;
n_fft = param.n_fft;

N = size(database_test,1);
n_class = max(database_test(:,end-1));

prior_g = zeros(1,n_class);
for k = 1:n_class
    prior_g(k) = sum(database_test(:,end-1) == k);
end
prior_g = log( prior_g ./ sum(prior_g) );

L = zeros(1,n_class);
posterior_g = zeros(N, n_class);
computation_time = zeros(1,N);
for b = 1:N
    tic
    data = database_test(b,1:end-2);
    if 20*log10(sum(abs(data).^2)) < threshold
        posterior_g(b,:) = NaN;
        continue;
    else
        data = database_test(b,1:end-2);
        
        spectrum = abs(fft(data)).^power;
        spectrum_norm = spectrum(1:n_fft) ./ sum(spectrum(1:n_fft));
        
        for k = 1:n_class
            L(k) = LSE(mixture_dirichletpdf(spectrum_norm, model{k}.alpha, model{k}.prop)');
        end
        
        posterior_g(b,:) = LSE_normprob(L + prior_g);
    end
    computation_time(b) = toc;
    progressbar(b,N)
end