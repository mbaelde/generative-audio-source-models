function prop_em = em_algo_prop(mixture_pdf, feature_norm, n_try, iter_max, tol, verbose)

[M,N_spect] = size(feature_norm);

BIC = zeros(1,n_try);
mprop_em = zeros(M,n_try);
for ntry = 1:n_try
    if verbose
        fprintf('--- try: %2.0f ---\n',ntry);
    end
    prop_em = rand(M,1);
    prop_em = prop_em ./ sum(prop_em);

    L = zeros(1,iter_max);
    iter_cur = 1;
    stop = false;
    while ~stop
        % E-step
        expectation = zeros(M,N_spect);
        for n = 1:N_spect
            expectation(:,n) = prop_em .* feature_norm(:,n);
            expectation(:,n) = expectation(:,n) ./ sum(expectation(:,n));
        end
        % M-step
        prop_em = sum(repmat(mixture_pdf,[M,1]) .* expectation,2) ./ sum(mixture_pdf);
        L(iter_cur) = sum(mixture_pdf .* log( sum(repmat(prop_em,[1,N_spect]) .* feature_norm) ));
        if verbose
            if iter_cur == 1
            	fprintf('iter: %2.0f ; L: %5.4f ; diff : inf \n', iter_cur, L(iter_cur));
            else
                fprintf('iter: %2.0f ; L: %5.4f ; diff : %1.6f \n', iter_cur, L(iter_cur), L(iter_cur) - L(iter_cur-1));
            end
        end
        iter_cur = iter_cur + 1;
        if iter_cur == 2
            stop = (iter_cur > iter_max) || (abs(L(iter_cur-1)) < tol);
        else
            stop = (iter_cur > iter_max) || (abs(L(iter_cur-1) - L(iter_cur-2)) < tol);
        end
        
    end
    BIC(ntry) = -2*L(iter_max) + M*log(N_spect);
    if verbose
        fprintf('BIC:  %5.4f \n', BIC(ntry));
    end
    mprop_em(:,ntry) = prop_em;
end
prop_em = mprop_em(:,find(BIC==min(BIC),1));