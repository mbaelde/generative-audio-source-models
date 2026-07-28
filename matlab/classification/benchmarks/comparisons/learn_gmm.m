function model = learn_gmm(feature_training, param)

n_class = max(feature_training(:,end-1));

M_max = param.M_max;
try_max = param.try_max;
best_model = cell(1,n_class);

for k = 1:n_class
    feature_class = feature_training(feature_training(:,end-1) == k, 1:end-2);
    model = cell(1,M_max);
    for M = 1:M_max
        clc
        disp(['--- Class: ',num2str(k),' ---'])
        disp(['M: ', num2str(M), ' / ',num2str(M_max)])
        param.M = M;
        for ntry = 1:try_max
            try
                model{M} = fitgmdist(feature_class, M,'RegularizationValue',0.0001);
                break;
            catch
                clc
                disp(['--- Class: ',num2str(k),' ---'])
                disp(['M: ', num2str(M), ' / ',num2str(M_max)])
                disp(['-Retry: ', num2str(ntry), ' / ',num2str(try_max)])
            end
        end
        if ntry == try_max
            disp(['Unable to converge for M = ',num2str(M)])
            model{M}.prop = [];
            model{M}.mu = [];
            model{M}.sigma = [];
            model{M}.BIC = Inf;
        end
        %save('temp_model_gmm.mat','model')
    end
    BIC = zeros(1,M_max);
    for M = 1:M_max
        BIC(M) = model{M}.BIC;
    end
    best_model{k} = model{find(BIC == min(BIC),1)};
    %save('temp_best_model_gmm.mat','best_model')
end
model = best_model;