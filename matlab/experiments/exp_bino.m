n = 513;
p_1 = 0.4;
p_2 = 0.5;
N = 1e4;
x_1 = binornd(n,p_1,[N,1]);
x_2 = binornd(n,p_2,[N,1]);
y = x_1+x_2;

p_y = (p_1+p_2)/2;
p_y1 = 1/2 - (2*p_1^2 - 2*p_1 + 2*p_2^2 - 2*p_2 + 1)^(1/2)/2;
p_y2 = 1/2 + (2*p_1^2 - 2*p_1 + 2*p_2^2 - 2*p_2 + 1)^(1/2)/2;

y_theo = binornd(2*n,p_y,[N,1]);

mean(y)
mean(y_theo)
var(y)
var(y_theo)
pchip
h = figure(1);
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
clf
histogram(y,20)
hold on
histogram(y_theo,20)
legend('Empirical','Theoretical')
title('X_1 \sim Bin(n,p_1) and X_2 \sim Bin(n,p_2), Y = X_1 + X_2 [Empirical] and Y \sim Bin(2n,(p_1+p_2)/2) [Theoretical]')


phat = binofit(y,2*n);
mean(phat)
%%
n = 20;
p_1 = 0.4;
N = 1e4;
x_1 = binornd(n, p_1, [N,1]);

q = 100;
x_q = q * x_1;

x_theo = binornd(n*q, p_1, [N,1]);

mean(x_q)
mean(x_theo)

h = figure(1);
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
clf
histogram(x_q,20)
hold on
histogram(x_theo,20)
legend('Empirical','Theoretical')
title('X_1 \sim Bin(n,p_1) and X_2 \sim Bin(n,p_2), Y = X_1 + X_2 [Empirical] and Y \sim Bin(2n,(p_1+p_2)/2) [Theoretical]')


%%
syms x y p_1 p_2 q n p

p_x1 = nchoosek(n,x) * p_1^(x) * (1-p_1)^(n-x);
p_x2 = nchoosek(n,y-x) * p_2^(y-x) * (1-p_2)^(n-y+x);

p_px1 = simplify(nchoosek(n,x/p) * p_1^(x/p) * (1-p_1)^(n-x/p));
p_qx2 = simplify(nchoosek(n,(y-x)/(q-p)) * p_2^((y-x)/(q-p)) * (1-p_2)^(n-(y-x)/(q-p)));
