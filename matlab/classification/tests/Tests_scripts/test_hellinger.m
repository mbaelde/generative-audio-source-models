addpath(genpath('Statistics'))
%%
mu = [0,2];
sigma = [0.5,0.5];
p = [0.5,0.5];

x = -2:0.01:4;

f = mixture_normpdf(x,mu,sigma,p);

figure(1)
clf
plot(x,f)

sqrt_f = sqrt(f);

figure(1)
hold on
plot(x,sqrt_f);
legend('f','\sqrt{f}')