clear
addpath(genpath(pwd))

fs = 44100;
%% Model 1
p_1 = [0.2,0.4,0.4];
mu_1 = [200,3000,6000];
sigma_1 = [20,30,40].^2;

x = 0:fs/1024:204*fs/1024;

pdf_1 = mixture_normpdf(x, mu_1, sigma_1, p_1);

figure(1)
clf
plot(x,pdf_1)

%% Model 2
p_2 = [0.4,0.2,0.2,0.2];
mu_2 = [500,2000,4000,7000];
sigma_2 = [10,20,30,40].^2;

x = 0:fs/1024:204*fs/1024;

pdf_2 = mixture_normpdf(x, mu_2, sigma_2, p_2);

figure(2)
clf
plot(x,pdf_2)

%% Use Monte Carlo method to approximate KL divergence
%N = 5;
n_try = 10;
for N = 0:6
    for n = 1:n_try
        tic
        x_mc = mixture_randn(10^N,mu_1,sigma_1,p_1);
        D_KL(N+1,n) = mean(log(mixture_normpdf(x_mc, mu_1, sigma_1, p_1) ./ mixture_normpdf(x_mc, mu_2, sigma_2, p_2)));
        elapsed_mc(N+1,n) = toc;
%         tic
%         D_s(N+1,n) = kl_variational_gmm_mex(p_1, mu_1, sigma_1, p_2, mu_2, sigma_2);
%         elapsed_var(N+1,n) = toc;
    end
    disp('------------')
    disp(['N: 10^',num2str(N)])
    disp(['mean KL (MC): ',num2str(mean(D_KL(N+1,:)))])
    disp(['std KL (MC): ',num2str(std(D_KL(N+1,:)))])
    disp(['time (MC): ',num2str(mean(elapsed_mc(N+1,:)))])
%     disp(['mean KL (Var): ',num2str(mean(D_s(N+1,:)))])
%     disp(['std KL (Var): ',num2str(std(D_s(N+1,:)))])
%     disp(['time (Var): ',num2str(mean(elapsed_var(N+1,:)))])
end
