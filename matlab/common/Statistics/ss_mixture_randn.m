function x_set = ss_mixture_randn(N, mu, sigma, mixture_weight, C_pre, P_label, P_class)
% -- Inputs --
% N: number of samples
% mu: means of gaussians
% sigma: standard deviation of gaussians
% mixture_weight: mixture coefficients
% C_pre: set of predefined components, i.e. components that generate both
% labeled and unlabeled samples.

% M_j: component selected to generate a sample

% -- Outputs --
% x_set, comprising:
% - x: the sample
% - c: the label, equals 1 to length(P_class), and 0 if no label
% - nu: equal 1 if there is a label, 0 else

% Number of mixture components
M = length(mixture_weight);
% Number of class
num_class = length(P_class);
% Set of random observations
x_labeled = [];
x_unlabeled = [];

for n = 1:N
    % Select a component and draw a sample form this component
    u = rand(1);
    if u <= mixture_weight(1)
        x = mu(1) + sigma(1)*randn(1);
        M_j = 1;
    else
        for m = 2:M
            if (sum(mixture_weight(1:m-1)) < u) && (u <= sum(mixture_weight(1:m)))
                x = mu(m) + sigma(m)*randn(1);
                M_j = m;
            end
        end
    end
    
    % If this component is a predefined component
    if any(M_j == C_pre)
        % Draw nu, that is if the sample has a label or not, according to
        % P_label
        u = rand(1);
        if u <= P_label(1)
            nu = 1;
        else
            nu = 0;
            x_unlabeled = [x_unlabeled; x, 0];
        end
        % If the sample has a label
        if nu
            % Select a label according to P_class
            u = rand(1);
            if u <= P_class(1)
                x_labeled = [x_labeled; x, 1, 1];
            else
                for k = 2:num_class
                    if (sum(P_class(1:k-1)) < u) && (u <= sum(P_class(1:k)))
                        x_labeled = [x_labeled; x, k, 1];
                    end
                end
            end
        end
    else
        x_unlabeled = [x_unlabeled; x, 0];
    end
end

x_set = { x_labeled, x_unlabeled };