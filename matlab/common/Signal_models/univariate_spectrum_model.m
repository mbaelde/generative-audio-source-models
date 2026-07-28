function model = univariate_spectrum_model(spectrum, fs, N_spect, M_set, f, verbose, criterion)
%% spectrum_model
%
% This function returns a mixture model interval-based modelizing the energy spectrum. It
% takes the temporal data sampled at fs, and model the N_spect-points
% spectrum. M_set represents the set of candidates number of mixture 
% components, f is a vector of frequencies, verbose is a flag to display
% information and criterion is either 'AIC' or 'BIC'.
%
% model = spectrum_model(data, fs, N_spect, M_set, f, verbose, criterion)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

if size(spectrum,1) > size(spectrum,2)
    spectrum = spectrum';
end

% EM algorithm parameters
epsilon = 1e-3;                     % Tolerance for stopping criteria
iter_max = 100;                     % Maximum number of iterations
count_max = 150;                     % Maximum number of retries when the 
                                    % algorithm didnt converge
% Initialize some variables
if strcmp(criterion,'AIC')
    AIC = zeros(length(M_set),1);
elseif strcmp(criterion,'BIC')
    BIC = zeros(length(M_set),1);
end
aux_model = cell(size(M_set));

% Compute the FFT of the data and take the first N_spect points of the
% energy spectrum
spectrum_norm = N_spect * spectrum(1:N_spect) ./ sum( spectrum(1:N_spect) );

% Main loop: test each value M in M_set and return the best model according
% to criterion

parfor m = 1:length(M_set)
    M = M_set(m);
    rspectrum_norm = repmat(spectrum_norm,[M,1]);
    if verbose
        disp(['Test for M = ', num2str(M)])
    end
    ok = 0;         % Variable that control the convergence of the EM algorithm
    count = 0;      % Current trial
    while (~ok) && (count < count_max)
        % Initialization
        mu = f(sort(randperm(N_spect, M)));     % Means of gaussians
        sigma = (fs / 100)^2 * rand(M,1);       % Variance of gaussians
        p_w = (1/M) * ones(M,1);                % Mixing coefficients
        
        % EM Algorithm
        stop = 0;
        iter = 1;
        L = zeros(iter_max,1);                  % Vector containing the log-likelihood
        L(1) = -Inf;
        while ~stop
            mu_old = mu;
            p_old = p_w;
            % E-step
            H_1 = zeros(M, N_spect);
            H_2 = zeros(M, N_spect);
            H_3 = zeros(M, N_spect);
            for m_aux = 1:M
                mypdf = normpdf(f, mu(m_aux), sqrt(sigma(m_aux)));
                H_1(m_aux,:) = diff( normcdf(f, mu(m_aux), sqrt(sigma(m_aux))) );
                H_2(m_aux,:) = diff( mypdf );
                H_3(m_aux,:) = diff( f .* mypdf );
            end
            
            %G = zeros(3, M, N_spect);
            G_1 = H_1;
            G_2 = repmat(mu,[1,N_spect]) .* H_1 - repmat(sigma,[1,N_spect]) .* H_2;
            
            rp_w = repmat(p_w,[1,N_spect]);
            mycdf = repmat(diff( cmixture_uninormcdf(f', mu, sqrt(sigma), p_w)),[M,1]);
            A_1 = rp_w .* G_1 ./ mycdf;
            A_2 = rp_w .* G_2 ./ mycdf;
            
            % M-step
            p_w = sum(rspectrum_norm .* A_1,2) ./ sum(spectrum_norm);
            evidence = sum(rspectrum_norm .* A_1, 2);
            mu = sum(rspectrum_norm .* A_2,2) ./ evidence;
            
            G_3 = repmat(sigma,[1,N_spect]) .* (H_1 + repmat(2.*mu - mu_old,[1,N_spect]) .* H_2 - H_3) + repmat((mu - mu_old).^2,[1,N_spect]) .* H_1;
            %A_3 = repmat(p_old,[1,N_spect]) .* G_3 ./ repmat(diff( cmixture_uninormcdf(f', mu, sqrt(sigma), p_w)),[M,1]);
            A_3 = repmat(p_old,[1,N_spect]) .* G_3 ./ mycdf;
            sigma = sum(rspectrum_norm .* A_3,2) ./ evidence;
            
            % Stopping criterion
            iter = iter + 1;
            L(iter) = sum(spectrum_norm .* log(diff( cmixture_uninormcdf(f', mu, sqrt(sigma), p_w))));
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
    k = 3*M-1; % number of parameters to estimate
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