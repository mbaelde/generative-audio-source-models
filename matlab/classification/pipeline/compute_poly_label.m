function likelihood_group = compute_poly_label(mixture_spectrum, feature_reduced, estimate_phiFlag, prop_true, verbose_g)

n_class = max(feature_reduced(:,end-1));

model_size = zeros(1,n_class);
for k = 1:n_class
   model_size(k) = sum(feature_reduced(:,end-1) == k);
end

M = nchoosek(n_class,1) + nchoosek(n_class,2);

likelihood_group = zeros(1,M);
for k = 1:n_class
    feature_class = feature_reduced(feature_reduced(:,end-1)==k,:);
    rmixture_spectrum = repmat(mixture_spectrum,[size(feature_class,1),1]);
    L = sum(rmixture_spectrum .* log(feature_class(:,1:end-2)),2);
    likelihood_group(k) = LSE(L');
end

n_try = 1;
iter_max = 20;
tol = 1e-3;
verbose = 0;
offset = n_class+1;

for k1 = 1:n_class-1
    if estimate_phiFlag  
        feature_1 = prop(1) * feature_reduced(feature_reduced(:,end-1)==k1,:);
    else
        feature_1 = feature_reduced(feature_reduced(:,end-1)==k1,:);
    end
    for k2 = k1+1:n_class
        if estimate_phiFlag  
            feature_2 = prop(2) * feature_reduced(feature_reduced(:,end-1)==k2,:);
        else
            feature_2 = feature_reduced(feature_reduced(:,end-1)==k2,:);
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
                    prop = em_algo_prop(mixture_spectrum, [model_1;model_2], n_try, iter_max, tol, verbose);
                    mixt_model = prop(1) * model_1 + prop(2) * model_2;
                    L(cnt) = sum(mixture_spectrum .* log(mixt_model));
                    cnt = cnt + 1;
                end
            end
            likelihood_group(offset) = LSE(L) - log(prod(model_size([k1,k2])));
            offset = offset + 1; 
        else
            prop = prop_true;
%             idx_feat = repmat(1:model_size(k1),[model_size(k2),1]);
%             mixt_model = feature_1(idx_feat(:),:) + repmat(feature_2,[prod(model_size([k1,k2])),1]);
%             L = sum(repmat(mixture_spectrum,[prod(model_size([k1,k2])),1]) .* log(mixt_model),2);
            cnt = 0;
            L = zeros(1,prod(model_size([k1,k2])));
            for n1 = 1:model_size(k1)
                model_1 = repmat(feature_1(n1,1:end-2),[model_size(k2),1]);
                model_2 = feature_2(:,1:end-2);
                mixt_model = model_1 + model_2;
                
                L(cnt+1:cnt+model_size(k2)) = sum(repmat(mixture_spectrum,[model_size(k2),1]) .* log(mixt_model),2);
                cnt = cnt + model_size(k2);
            end
            likelihood_group(offset) = LSE(L) - log(prod(model_size([k1,k2])));
            offset = offset + 1;
        end
    end
end
