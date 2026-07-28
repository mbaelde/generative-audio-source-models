function model = learn_dmm(feature_training, param)

n_class = max(feature_training(:,end-1));

learning_rate = param.learning_rate;
M_max = param.M_max;
try_max = param.try_max;
best_model = cell(1,n_class);

%M = M_max;

for k = 1:n_class
    feature_class = feature_training(feature_training(:,end-1) == k, 1:end-2);
    model = cell(1,length(M_max));
    for M = M_max
        clc
        disp(['--- Class: ',num2str(k),' ---'])
        disp(['M: ', num2str(M), ' / ',num2str(length(M_max))])
        param.M = M;
        param.alpha_factor = 0.1 / (log(M)*(M>1) + M*(M==1));
        for ntry = 1:try_max
            [prop_em, alpha, L] = fit_mixture_dirichlet(feature_class, param);
            if ~isnan(L(end))
                model{M}.prop = prop_em;
                model{M}.alpha = alpha;
                model{M}.AIC = -2*L(end) + 2*M*param.n_fft;
                model{M}.BIC = -2*L(end) + M*param.n_fft*log(size(feature_class,1));
                model{M}.L = L(end);
                break;
            else
                clc
                disp(['--- Class: ',num2str(k),' ---'])
                disp(['M: ', num2str(M), ' / ',num2str(M_max)])
                disp(['-Retry: ', num2str(ntry), ' / ',num2str(try_max)])
                param.alpha_factor = learning_rate * param.alpha_factor;
            end
        end
        if ntry == try_max
            disp(['Unable to converge for M = ',num2str(M)])
            model{M}.prop = [];
            model{M}.alpha = [];
            model{M}.AIC = 0;
            model{M}.BIC = 0;
            model{M}.L = 0;
        end
        save(['temp_model_n',num2str(param.n_idx),'_f',num2str(param.fold),'.mat'],'model')
    end
    
    AIC = zeros(1,length(M_max));
    BIC = zeros(1,length(M_max));
    L = zeros(1,length(M_max));
    for M = M_max
        AIC(M) = model{M}.AIC;
        BIC(M) = model{M}.BIC;
        L(M) = model{M}.L;
    end
    
    idx_zero = L == 0;
    AIC(idx_zero) = nan;
    BIC(idx_zero) = nan;
    L(idx_zero) = nan;
        
    best_model{k} = model{find(AIC == min(AIC),1)};
%    best_model{k} = model{M};
    save(['temp_best_model_n',num2str(param.n_idx),'_f',num2str(param.fold),'.mat'],'best_model')
end
model = best_model;