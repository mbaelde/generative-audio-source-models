function likelihood_group = compute_poly_label2(mixture_spectrum, feature_reduced, estimate_phiFlag, prop_true, verbose_g)

n_fft = length(mixture_spectrum);
n_class = max(feature_reduced(:,end-1));

model_size = zeros(1,n_class);
for k = 1:n_class
   model_size(k) = sum(feature_reduced(:,end-1) == k);
end
cum_model_size = cumsum(model_size);
msize = [0, cum_model_size];

M = nchoosek(n_class,1) + nchoosek(n_class,2);

likelihood_group = zeros(1,M);
% Monoclass
rmixture_spectrum = repmat(mixture_spectrum, [sum(model_size),1]);
L = sum(rmixture_spectrum .* log(feature_reduced(:,1:end-2)),2);
for k = 1:n_class
    A = L(msize(k)+1:msize(k+1))';
    likelihood_group(k) = LSE(A);
end
likelihood_group(1:n_class) = likelihood_group(1:n_class) - log(model_size);

% Multiclass
% n_try = 1;
% iter_max = 20;
% tol = 1e-3;
% verbose = 0;
offset = n_class+1;

q_phi = 100;
p_phi = (1:(q_phi-1))';
rphi = (repmat(p_phi/q_phi,[1,n_fft]));
rqmix = repmat(q_phi*mixture_spectrum,[q_phi-1,1]);
for k1 = 1:n_class-1
    if estimate_phiFlag  
        feature_1 = feature_reduced(feature_reduced(:,end-1)==k1,:);
    else
        prop_est = prop_true;
        feature_1 = prop_est(1) * feature_reduced(feature_reduced(:,end-1)==k1,1:end-2);
    end
    for k2 = k1+1:n_class
        if estimate_phiFlag  
            feature_2 = feature_reduced(feature_reduced(:,end-1)==k2,:);
        else
            feature_2 = prop_est(2) *  feature_reduced(feature_reduced(:,end-1)==k2,1:end-2);
        end
        if verbose_g
            fprintf('(k1,k2): (%f,%f)\n', k1, k2);
        end
        if estimate_phiFlag  
            cnt = 1;
            L = zeros(1,prod(model_size([k1,k2])));
            for n1 = 1:model_size(k1)
                model_1 = feature_1(n1,1:end-2);
                for n2 = 1:model_size(k2)
                    model_2 = feature_2(n2,1:end-2);
                    %prop = em_algo_prop_mex(mixture_spectrum, [model_1;model_2], n_try, iter_max, tol, verbose);
                    
                    mixt_model = rphi .* repmat(model_1,[q_phi-1,1]) + (1-rphi) .* repmat(model_2,[q_phi-1,1]);
                    aux_L = sum(rqmix .* log(mixt_model),2);
                    [~,idx_max] = max(aux_L);
                    
                    prop_est = p_phi(idx_max) / q_phi;
                    mixt_model = prop_est * model_1 + (1-prop_est) * model_2;
                    L(cnt) = sum(mixture_spectrum .* log(mixt_model));
                    cnt = cnt + 1;
                end
            end
            likelihood_group(offset) = LSE(L) - log(prod(model_size([k1,k2])));
            offset = offset + 1; 
        else   
            cnt = 0;
            L = zeros(1,prod(model_size([k1,k2])));
            if model_size(k1) < model_size(k2)
                rmixture_spectrum = repmat(mixture_spectrum,[model_size(k2),1]);
                for n1 = 1:model_size(k1)
                    mixt_model = repmat(feature_1(n1,:),[model_size(k2),1]) + feature_2;

                    L(cnt+1:cnt+model_size(k2)) = sum(rmixture_spectrum .* log(mixt_model),2);
                    cnt = cnt + model_size(k2);
                end
            else
                rmixture_spectrum = repmat(mixture_spectrum,[model_size(k1),1]);
                for n2 = 1:model_size(k2)
                    mixt_model = feature_1 + repmat(feature_2(n2,:),[model_size(k1),1]);

                    L(cnt+1:cnt+model_size(k1)) = sum(rmixture_spectrum .* log(mixt_model),2);
                    cnt = cnt + model_size(k1);
                end
            end
            likelihood_group(offset) = LSE(L) - log(prod(model_size([k1,k2])));
            offset = offset + 1;
        end
    end
end
