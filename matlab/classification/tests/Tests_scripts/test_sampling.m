clear
addpath('Statistics')
%% Inverse CDF
% 1 - Generate u \sim U([0,1])
N = 10000;
u = rand(N,1);
lambda = 2;
% 2 - Compute inverse cdf of u
x = (1/lambda) * log(1./(1-u));

% Compute histogram
[h,c] = hist(x,21);
delta = c(2)-c(1);
h = h ./ sum(h) / delta;
% Compute theoretical pdf
pdf = lambda * exp(-lambda .* (0:0.01:5));
pdf = pdf ./ max(pdf);
% Plot results
figure(1)
clf
bar(c,h)
hold on
plot((0:0.01:5),pdf)
xlabel('x')
ylabel('probability')
title('Samples from an exponentional distribution using inverse cdf method')

%% Accep-Reject
N = 10000;
M = 5;
max_iter = 100;
for n = 1:N
    cur_iter = 1;
    % 1 - Generate u \sim U([0,1]
    u = rand(1);
    % 2 - Generate x \sim g(x)
    aux = trnd(1,1,1);
    % 3 - Stopping criterion
    stop = (normpdf(aux) / (M * cauchypdf(aux,1))) > u;
    while ~stop
        % 1 - Generate u \sim U([0,1]
        u = rand(1);
        % 2 - Generate x \sim g(x)
        aux = trnd(1,1,1);
        cur_iter = cur_iter + 1;
        % 3 - Stopping criterion
        stop = ((normpdf(aux) / (M * cauchypdf(aux,1))) > u) | (cur_iter > max_iter);
    end
    x(n) = aux;
end

[h,c] = hist(x,21);
delta = c(2)-c(1);
h = h ./ sum(h) / delta;

pdf = normpdf(-5:0.01:5);
pdf = pdf ./ max(pdf);

figure(1)
clf
bar(c,h)
hold on
plot(-5:0.01:5,pdf)
xlabel('x')
ylabel('probability')
title('Samples from an normal distribution using accept-reject method')

%% Metropolis-Hasting
N = 20000;
t0 = 5000;
a = -20;
b = 20;

x_c = [];
t = 1;
x = 0;
while length(x_c) < N
    y = (b-a)*rand(1)+a;
    p = min(1, normpdf(y)/normpdf(x));
    if rand(1) < p
        x = y;
    end
    if t > t0
        x_c = [x_c, x];
    end
    t = t + 1;
end

[h,c] = hist(x_c,21);
delta = c(2)-c(1);
h = h ./ sum(h) / delta;

pdf = normpdf(-5:0.01:5);
pdf = pdf ./ max(pdf);

figure(1)
clf
bar(c,h)
hold on
plot(-5:0.01:5,pdf)
xlabel('x')
ylabel('probability')
title('Samples from an normal distribution using Metropolis-Hasting method')

%% Truncated normal variable
mu = 1;
sigma = 2;

mu_m = -4;
mu_p = 1;

N = 10000;

for n = 1:N
    x(n) = univariate_truncated_randn(mu, mu_m, mu_p, sigma);
end

[h,c] = hist(x,21);
delta = c(2)-c(1);
h = h ./ sum(h) / delta;

pdf = normpdf(-10:0.01:10, mu, sqrt(sigma));
pdf = pdf ./ max(pdf);

figure(1)
clf
bar(c,h)
hold on
plot(-10:0.01:10,pdf)
xlabel('x')
ylabel('probability')
title('Samples from an normal distribution using Metropolis-Hasting method')

%%
% mu = [1,5];
% sigma = [0.5,2];
% p = [0.7,0.3];
% N = 10000;
% 
% for n = 1:N
%     u = rand(1);
%     if u < p(1)
%         x(n) = mu(1) + sqrt(sigma(1))*randn(1);
%     end
%     for k = 2:length(p)
%         if (u > sum(p(1:k-1))) && (u <= sum(p(1:k)))
%             x(n) = mu(k) + sqrt(sigma(k))*randn(1);
%         end
%     end
% end
% x_v = -5:0.01:12;

mu = 5;
sigma = 2;

x = mu + sqrt(sigma) * randn(N,1);
x_v = -2:0.01:12;

[h,c] = hist(x,21);
delta = c(2)-c(1);
h = h ./ sum(h) / delta;

%pdf = mixture_normpdf(-5:0.01:12, mu, sqrt(sigma),p);
pdf = normpdf(x_v, mu, sqrt(sigma));
%pdf = pdf ./ max(pdf);

figure(1)
clf
bar(c,h)
hold on
plot(x_v,pdf)
xlabel('x')
ylabel('probability')
legend('Normalized Histogram', 'Fitted pdf')
