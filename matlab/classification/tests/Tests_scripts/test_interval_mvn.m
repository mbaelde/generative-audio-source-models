clear
addpath(genpath('Statistics'))
linkage
%%
x_min = -6;
x_max = 6;
x_grid = 0.1;

y_min = -6;
y_max = 6;
y_grid = 0.1;

[X,Y] = meshgrid((x_min:x_grid:x_max), (y_min:y_grid:y_max));

%% Synthetic data
mu = [-2,-2;
    2, 2;
    3,-4];

sigma(:,:,1) = eye(2);
sigma(:,:,2) = [ 1  ,-0.1;
    -0.1, 1  ];
sigma(:,:,3) = eye(2)/3;
prop = [0.4; 0.3; 0.3];

N = 2000;

samples = mixture_mvnrnd(N,mu,sigma,prop);

figure(1)
clf
plot(samples(:,1),samples(:,2),'*')

nbins = [10,10];
figure(2)
clf
hist3(samples, nbins)
[H,C] = hist3(samples, nbins);

H_obs = H(:)';

%% Real data
load('F:\Archives\IMPACT-Memoire\Manip\Data\hrtfLMedianFront.mat')

hrtf = round(squeeze(hrtfLMedianFront(1,:,:))*10);
elev = -45:5:90;
freq = 0:44100/256:128*44100/256;

[D,F] = meshgrid(elev,freq);

figure(100)
surf(D,F,hrtf')
%%
H_obs = hrtf(:)';
Cx = elev;
Cy = freq;
%%
n_runs = 4;

params.verbose = 1;
params.count_max = 10;
params.iter_max = 100;

K = 2:2:10;

model = cell(1,n_runs);
criterion = zeros(1,n_runs);
for run = 1:n_runs
    disp([' #### Run number: ', num2str(run),' #### '])
    model{run} = gmm2d_binned(H_obs, Cx, Cy, K, params);
    criterion(run) = model{run}.likelihood;
end

id_best = criterion == max(criterion);
best_model = model{id_best};
%%
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


