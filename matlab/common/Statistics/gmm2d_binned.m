function model = gmm2d_binned(H_obs, Cx, Cy, K, params)
%% gmm2d_binned
%
% This function fits a gaussian mixture model on a 2D observed binned data 
% H_obs. Cx and Cy contains the discretization of the axes x and y and K
% the possible values for the number of components.
%
% model = gmm2d_binned(H_obs, Cx, Cy, K, params)
%
% Returns a structure model containing the means mu, the covariance matrix
% sigma, the mixing proportions prop and the fitted likelihood.
%
% Reference: Fitting Mixture Models to Binned Data (Chapter 9), Finite
% Mixture Models, G. McLachlan, Wiley, 2000.
%
% Author: Maxime Baelde
% A-Volute // 2016

% Set global variables used in he handle functions
global mu_em sigma_em k
% Verbosity
verbose = params.verbose;
% BIC and models variables
BIC = zeros(size(K));
models = cell(size(K));
% Total number of observations
N = sum(H_obs);
% Normalization constant for the likelihood computation
C_1 = sum(log(1:N));
for rr = 1:length(H_obs)
    aux = sum(log(1:H_obs(rr)));
    C_1 = C_1 - aux;
end
% Handles to the function used in the integral computation
fun_x = @compute_A1x;
fun_y = @compute_A1y;

fun_xx = @compute_A2xx;
fun_xy = @compute_A2xy;
fun_yx = @compute_A2yx;
fun_yy = @compute_A2yy;
%%%%% Loop over the different values in K %%%%%
for kk = 1:length(K)
    if verbose
        disp(['--- Fitting model with ', num2str(K(kk)), ' components ---'])
    end
    %%%%% Repeat count_max time the EM algorithm and keep the best result %%%%%
    count_max = params.count_max;
    ok = 0;
    count = 1;
    while (~ok) && (count < count_max)
        if verbose
            disp(['Trial number: ', num2str(count)])
        end
        % Set the seed
        rng('shuffle')
        % Dimension: 2D
        d = 2;
        % Create the grid
        delta_x = Cx(2)-Cx(1);
        delta_y = Cy(2)-Cy(1);
        [X_obs, Y_obs] = meshgrid([Cx,Cx(end)+delta_x], [Cy,Cy(end)+delta_y]);
        X_obs = X_obs - delta_x/2;
        Y_obs = Y_obs - delta_y/2;
        % Number of elements in the grid
        nx = size(X_obs,2)-1;
        ny = size(Y_obs,1)-1;
        r = nx * ny;
        % Initialisation
        mu_em = zeros(K(kk),d);
        sigma_em = zeros(d,d,K(kk));
        for k = 1:K(kk)
            u = rand(1);
            mu_em(k,:) = (-1 * (u > 0.5) + 1 * (u <= 0.5)) * randn(1,d);
            sigma_em(:,:,k) = randn(d,d,1);
            sigma_em(:,:,k) = sigma_em(:,:,k)'*sigma_em(:,:,k);
        end
        prop_em = ones(K(kk),1) ./ K(kk);
        % Compute the probability of the model in each cell
        cnt = 1;
        P_r = zeros(1,r);
        for i = 1:ny
            for j = 1:nx
                bottom = [X_obs(i,j), Y_obs(i,j)];
                up = [X_obs(i+1,j+1), Y_obs(i+1,j+1)];
                P_r(cnt) = mixture_mvncdf(bottom, up, mu_em, sigma_em, prop_em);
                cnt = cnt + 1;
            end
        end
        % EM parameters
        iter_max = params.iter_max;
        stop = 0;
        L = zeros(1,iter_max);
        L(1) = -Inf;
        iter_cur = 1;
        
        %%%%% Main loop %%%%%
        while ~stop
            prop_old = prop_em;
            %--- E-step ---%
            if any(P_r == 0)
                ok = 0;
                count = count + 1;
                if verbose
                    disp('Error: singularity')
                end
                break;
            end
            % Compute conditional expectations of the posterior
            % probabilities of the latent variable Z
            A_0 = zeros(K(kk),r);
            A_1 = zeros(K(kk),d,r);
            for k = 1:K(kk)
                cnt = 1;
                for i = 1:ny
                    for j = 1:nx
                        bottom = [X_obs(i,j), Y_obs(i,j)];
                        up = [X_obs(i+1,j+1), Y_obs(i+1,j+1)];
                        
                        A_0(k,cnt) = prop_em(k) * mvncdf(bottom, up, mu_em(k,:), sigma_em(:,:,k))/ P_r(cnt);
                        
                        A_1(k,1,cnt) = prop_em(k) * integral2(fun_x, bottom(1), up(1), bottom(2), up(2)) / P_r(cnt);
                        A_1(k,2,cnt) = prop_em(k) * integral2(fun_y, bottom(1), up(1), bottom(2), up(2)) / P_r(cnt);
                        
                        cnt = cnt + 1;
                    end
                end
            end
            
            %--- M-step ---%
            % Update the parameters
            c_psi = sum(repmat(H_obs,[K(kk),1]) .* A_0,2);
            
            for k = 1:K(kk)
                prop_em(k) = c_psi(k) / N;
                mu_em(k,:) = sum(repmat(reshape(H_obs,[1,1,r]),[1,d,1]) .* A_1(k,:,:),3) ./ c_psi(k);
            end
            
            A_2 = zeros(K(kk),d,d,r);
            for k = 1:K(kk)
                cnt = 1;
                for i = 1:ny
                    for j = 1:nx
                        bottom = [X_obs(i,j), Y_obs(i,j)];
                        up = [X_obs(i+1,j+1), Y_obs(i+1,j+1)];
                        
                        A_2(k,1,1,cnt) = prop_old(k) * integral2(fun_xx, bottom(1), up(1), bottom(2), up(2)) / P_r(cnt);
                        A_2(k,1,2,cnt) = prop_old(k) * integral2(fun_xy, bottom(1), up(1), bottom(2), up(2)) / P_r(cnt);
                        A_2(k,2,1,cnt) = prop_old(k) * integral2(fun_yx, bottom(1), up(1), bottom(2), up(2)) / P_r(cnt);
                        A_2(k,2,2,cnt) = prop_old(k) * integral2(fun_yy, bottom(1), up(1), bottom(2), up(2)) / P_r(cnt);
                        
                        cnt = cnt + 1;
                    end
                end
            end
            
            err = zeros(size(K));
            for k = 1:K(kk)
                sigma_em(:,:,k) = squeeze(sum(repmat(reshape(H_obs,[1,1,1,r]),[1,d,d,1]) .* A_2(k,:,:,:),4)) ./ c_psi(k);
                [~,err(k)] = cholcov(sigma_em(:,:,k),0);
            end

            if any(err ~= 0)
                ok = 0;
                count = count + 1;
                if verbose
                    disp('Error: sigma not positive definite')
                end
                break;
            end
            
            %--- Stopping criterion ---%
            iter_cur = iter_cur + 1;
            % Compute the probability of the model in each cell
            cnt = 1;
            P_r = zeros(r,1);
            for i = 1:ny
                for j = 1:nx
                    bottom = [X_obs(i,j), Y_obs(i,j)];
                    up = [X_obs(i+1,j+1), Y_obs(i+1,j+1)];
                    P_r(cnt) = mixture_mvncdf(bottom, up, mu_em, sigma_em, prop_em);
                    cnt = cnt + 1;
                end
            end
            % Compute the observed likelihood
            L(iter_cur) = sum(H_obs' .* log(P_r./sum(P_r))) + C_1;
            if verbose
                disp(['L: ',num2str(L(iter_cur)), ' ; iter: ', num2str(iter_cur)])
            end
            % Stop if the the maximum number of iterations is reached
            stop = (iter_cur > iter_max);
        end
        if stop
            ok = 1;
            if verbose
                disp(['Converged. Current likelihood: ', num2str(L(iter_cur))])
            end
        end
    end
    % Compute the BIC criterion and store the current model
    n_params = K(kk) * (1 + d + d^2) - 1;
    BIC(kk) = -2 * L(iter_cur) + n_params * log(N);
    models{kk} = struct('mu',mu_em,'sigma',sigma_em,'prop',prop_em,'likelihood',L(iter_cur));
end

% Select the best model according to the BIC
model = models{BIC == min(BIC)};

%% Functions used in the integral computations
function A_1x = compute_A1x(x,y)
global mu_em sigma_em k
A_1x = x .* reshape(mvnpdf([x(:), y(:)], mu_em(k,:), sigma_em(:,:,k)), size(x));

function A_1y = compute_A1y(x,y)
global mu_em sigma_em k
A_1y = y .* reshape(mvnpdf([x(:), y(:)], mu_em(k,:), sigma_em(:,:,k)), size(y));
   
function A_2xx = compute_A2xx(x,y)
global mu_em sigma_em k
A_2xx = (x - mu_em(k,1)).^2 .* reshape(mvnpdf([x(:), y(:)], mu_em(k,:), sigma_em(:,:,k)), size(x));

function A_2xy = compute_A2xy(x,y)
global mu_em sigma_em k
A_2xy = (x - mu_em(k,1)) .* (y - mu_em(k,2)) .* reshape(mvnpdf([x(:), y(:)], mu_em(k,:), sigma_em(:,:,k)), size(x));

function A_2yx = compute_A2yx(x,y)
global mu_em sigma_em k
A_2yx = (y - mu_em(k,2)) .* (x - mu_em(k,1)) .* reshape(mvnpdf([x(:), y(:)], mu_em(k,:), sigma_em(:,:,k)), size(y));
                
function A_2yy = compute_A2yy(x,y)
global mu_em sigma_em k          
A_2yy = (y - mu_em(k,2)).^2 .* reshape(mvnpdf([x(:), y(:)], mu_em(k,:), sigma_em(:,:,k)), size(y));