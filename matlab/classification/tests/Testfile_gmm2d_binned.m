clear
%% Define a sampling space using a meshgrid
x_min = -6;
x_max = 6;
x_grid = 0.1;

y_min = -6;
y_max = 6;
y_grid = 0.1;

[X,Y] = meshgrid((x_min:x_grid:x_max), (y_min:y_grid:y_max));

%% Synthetic data
% Set the random seed for reproducibility
rng(12345);
% Means of the mixture
mu = [-2,-2;
       2, 2;
       3,-4];
% Covariance matrices of the mixture
sigma(:,:,1) = eye(2);
sigma(:,:,2) = [ 1  ,-0.1;
                -0.1, 1  ];
sigma(:,:,3) = eye(2)/3;
% Mixing coefficients
prop = [0.4; 0.3; 0.3];
% Number of points to sample
N = 2000;
% Samples from a gaussian mixture model in 2D, with means mu, covariances
% matrices sigma and mixing coefficients prop
samples = mixture_mvnrnd(N,mu,sigma,prop);
% Display the samples in a plane
figure(1)
clf
plot(samples(:,1),samples(:,2),'*')
% Create an histogram from the samples: these are the gross data
% Binned the data into 10 x 10 boxes
nbins = [10,10];
figure(2)
clf
hist3(samples, nbins)
% H contains the number of occurences in each box, and C the grid in x and
% y
[H,C] = hist3(samples, nbins);
% H_obs is just a vector form of H
H_obs = H(:)';
Cx = C{1};
Cy = C{2};
%% Model fitting
% Number of runs of the algorithm, prevent from fitting a local optimum.
% Should be a multiple of the number of core in your processor (more
% optimal in parallel computing)
n_runs = 4; 
% Parameters of the algorithm
params.verbose = 1;
params.count_max = 10;
params.iter_max = 100;
% Number of components in the mixture to test
K = 1:10;
% Variables declarations
model = cell(1,n_runs);
criterion = zeros(1,n_runs);
% Fitting
parfor run = 1:n_runs
    disp([' #### Run number: ', num2str(run),' #### '])
    model{run} = gmm2d_binned(H_obs, Cx, Cy, K, params);
    % ### PUT HERE YOU MEX IMPLEMENTATION ###
    %model{run} = gmm2d_binned_mex(H_obs, Cx, Cy, K, params);
    criterion(run) = model{run}.likelihood;
end
% Select the best model according the likelihood
id_best = criterion == max(criterion);
best_model = model{id_best};
%% Display the best model
F = mixture_mvnpdf([X(:),Y(:)], best_model.mu, best_model.sigma, best_model.prop);
F = reshape(F,length(Y), length(X));
figure(1)
clf
plot(samples(:,1),samples(:,2),'*')
hold on
for k = 1:length(prop)
    plot(mu(k,1),mu(k,2),'+r','LineWidth',3,'MarkerSize',20)
end
contour(X,Y,F)
