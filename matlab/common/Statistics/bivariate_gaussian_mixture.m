function model = bivariate_gaussian_mixture(data, D_axis, F_axis, M_set, criterion)
%% bivariate_gaussian_mixture
%
% This function returns a mixture model of a bivariate histogram. M_set 
% represents the set of candidates number of mixture components, criterion 
% is either 'AIC' or 'BIC'.
% data is a matrix of data, from R^{D x F}
%
% model = bivariate_gaussian_mixture(data, D_axis, F_axis, M_set, criterion)
%
% Author: Maxime BAELDE
% Date: 03/2016
% Company: A-Volute / INRIA

[D,F] = size(data);
D_res = D_axis(2)-D_axis(1);
F_res = F_axis(2)-F_axis(1);

% EM algorithm parameters
epsilon = 1e-3;                     % Tolerance for stopping criteria
iter_max = 100;                     % Maximum number of iterations
count_max = 150;                     % Maximum number of retries when the 
                                    % algorithm didn't converge
% Initialize some variables
if strcmp(criterion,'AIC')
    AIC = zeros(length(M_set),1);
elseif strcmp(criterion,'BIC')
    BIC = zeros(length(M_set),1);
end
aux_model = cell(size(M_set));

% Normalize data so as it sums to D in the first dimension and to F in the
% second dimension.
data_norm = data;
data_norm = D * data_norm ./ repmat(sum(data_norm,1),[D,1]);
data_norm = F * data_norm ./ repmat(sum(data_norm,2),[1,F]);

% Main loop: test each value M in M_set and return the best model according
% to criterion

for m = 1:length(M_set)
    M = M_set(m);
    if verbose
        disp(['Test for M = ', num2str(M)])
    end
    ok = 0;         % Variable that control the convergence of the EM algorithm
    count = 0;      % Current trial
    while (~ok) && (count < count_max)
        % Initialization
        mu = [D_axis(randperm(D,M)), F_axis(randperm(F,M))];     % Means of gaussians
        sigma = repmat([D_res, F_res],[M,1]) .* rand(M,2);       % Variance of gaussians
        p_w = (1/M) * ones(M,1);                                 % Mixing coefficients
      
        % EM Algorithm
        stop = 0;
        iter = 1;
        L = zeros(iter_max,1);                                   % Vector containing the log-likelihood
        L(1) = -Inf;
        while ~stop
            mu_old = mu;
            p_old = p_w;
            % E-step
            HX_1 = zeros(M, D);
            HX_2 = zeros(M, D);
            HX_3 = zeros(M, D);
            HY_1 = zeros(M, F);
            HY_2 = zeros(M, F);
            HY_3 = zeros(M, F);
            for m_aux = 1:M
                mypdf_X = normpdf(D_axis, mu(m_aux,1), sqrt(sigma(m_aux,1)));
                mypdf_Y = normpdf(F_axis, mu(m_aux,2), sqrt(sigma(m_aux,2)));
                HX_1(m_aux,:) = diff( normcdf(D_axis, mu(m_aux,1), sqrt(sigma(m_aux,1))) );
                HY_1(m_aux,:) = diff( normcdf(F_axis, mu(m_aux,2), sqrt(sigma(m_aux,2))) );
                HX_2(m_aux,:) = diff( mypdf_X );
                HY_2(m_aux,:) = diff( mypdf_Y );
                HX_3(m_aux,:) = diff( D_axis .* mypdf_X );
                HY_3(m_aux,:) = diff( F_axis .* mypdf_Y );
            end
            
            GX_1 = HX_1;
            GY_1 = HY_1;
            GX_2 = repmat(mu(:,1),[1,D]) .* HX_1 - repmat(sigma(:,1),[1,D]) .* HX_2;
            GY_2 = repmat(mu(:,2),[1,F]) .* HY_1 - repmat(sigma(:,2),[1,F]) .* HY_2;
            
            mycdf_X = repmat(diff(mixture_normcdf(D_axis, mu(:,1), sqrt(sigma(:,1)), p_w)),[M,1]);
            mycdf_Y = repmat(diff(mixture_normcdf(F_axis, mu(:,2), sqrt(sigma(:,2)), p_w)),[M,1]);
            
            % M-step
            evidence = zeros(M,1);
            for m_aux = 1:M
                rp_w = p_w(m_aux) * ones(D,F);
                A_1 = rp_w .* ((GX_1(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* GY_1(m_aux,:))) ./ ( (mycdf_X(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* mycdf_Y(m_aux,:)) );
                evidence(m_aux) = sum(sum(data .* A_1));
                p_w(m_aux) = evidence(m_aux) ./ sum(data(:));
                A_21 = rp_w .* ((GX_2(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* GY_1(m_aux,:))) ./ ( (mycdf_X(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* mycdf_Y(m_aux,:)) );
                A_22 = rp_w .* ((GX_1(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* GY_2(m_aux,:))) ./ ( (mycdf_X(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* mycdf_Y(m_aux,:)) );
                mu(m_aux,1) = sum(sum(data .* A_21)) ./ evidence(m_aux);
                mu(m_aux,2) = sum(sum(data .* A_22)) ./ evidence(m_aux);
            end
            
            GX_31 = repmat(sigma(:,1),[1,D]) .* (HX_1 + repmat(2.*mu(:,1) - mu_old(:,1),[1,D]) .* HX_2 - HX_3) + repmat((mu(:,1) - mu_old(:,1)).^2,[1,D]) .* HX_1;
            GY_32 = repmat(sigma(:,2),[1,F]) .* (HY_1 + repmat(2.*mu(:,2) - mu_old(:,2),[1,F]) .* HY_2 - HY_3) + repmat((mu(:,2) - mu_old(:,2)).^2,[1,F]) .* HY_1;
            
            for m_aux = 1:M
                rp_w = p_old(m_aux) * ones(D,F);
                A_31 = rp_w .* ((GX_31(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* GY_1(m_aux,:))) ./ ( (mycdf_X(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* mycdf_Y(m_aux,:)) );
                A_32 = rp_w .* ((GY_32(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* GX_1(m_aux,:))) ./ ( (mycdf_X(m_aux,:)' .* ones(1,F)) .* (ones(D,1) .* mycdf_Y(m_aux,:)) );
                sigma(m_aux,1) = sum(sum(data_norm .* A_31)) / evidence(m_aux);
                sigma(m_aux,2) = sum(sum(data_norm .* A_32)) / evidence(m_aux);
            end
            
            % Stopping criterion
            iter = iter + 1;
            mycdf_X = diff(mixture_normcdf(D_axis, mu(:,1), sqrt(sigma(:,1)), p_w));
            mycdf_Y = diff(mixture_normcdf(F_axis, mu(:,2), sqrt(sigma(:,2)), p_w));
            L(iter) = sum(data .* log(diff(mycdf_X))) + sum(data .* log(diff(mycdf_Y)));
            if isnan(L(iter))
                stop = 1;
                ok = 0;
                if verbose
                    disp('Error: logL = NaN, relaunch EM algorithm with another initial values')
                end
                count = count + 1;
            else
                stop = ((abs(L(iter)-L(iter-1)) / abs(L(iter-1))) < epsilon) || iter > iter_max;
                ok = 1;
            end
            if verbose
                disp(['iter = ', num2str(iter), ' ; L = ', num2str(L(iter)), ' ; ok = ', num2str(ok), ' ; count = ', num2str(count), ' ; stop = ', num2str(stop)])
            end
        end
    end
    if (count == count_max) && verbose
        disp(['Not able to fit a model with M = ', num2str(M)])
    end
    
    aux_model{m}.n_components = M;
    aux_model{m}.mu = mu;
    aux_model{m}.sigma = sigma;
    aux_model{m}.mixing_coeff = p_w;
    aux_model{m}.iter = iter;
    aux_model{m}.logL = L(iter);
    
    % AIC/BIC evalutation
    k = 5*M-1; % number of parameters to estimate
    if strcmp(criterion,'AIC')
        AIC(m) = -2*L(iter) + 2*k;
    elseif strcmp(criterion,'BIC')
        BIC(m) = -2*L(iter) + log(N_spect)*k;
    end
end

% Return the best model
if strcmp(criterion,'AIC')
    model = aux_model{AIC == min(AIC)};
elseif strcmp(criterion,'BIC')
    model = aux_model{BIC == min(BIC)};
end