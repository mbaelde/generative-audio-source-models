addpath('Statistics')

alpha_1 = [2, 1, 1];            alpha_01 = sum(alpha_1);
alpha_2 = [1, 5, 1];            alpha_02 = sum(alpha_2);
alpha_sum = 0.5*(alpha_1 + alpha_2);  alpha_0sum = sum(alpha_sum);

mean_1 = alpha_1 ./ alpha_01;
mean_2 = alpha_2 ./ alpha_02;
mean_sum = alpha_sum ./ sum(alpha_sum);

var_1 = alpha_1 .* (alpha_01 - alpha_1) ./ (alpha_01^2 * (alpha_01 + 1));
var_2 = alpha_2 .* (alpha_02 - alpha_2) ./ (alpha_02^2 * (alpha_02 + 1));
var_sum = alpha_sum .* (alpha_0sum - alpha_sum) ./ (alpha_0sum^2 * (alpha_0sum + 1));

%%
x_1 = drchrnd(alpha_1,2000);
x_2 = drchrnd(alpha_2,2000);
x_sum = drchrnd(alpha_sum,2000);

emp_mean_1 = mean(x_1);
emp_mean_2 = mean(x_2);
emp_mean_sum = mean(x_sum);

emp_var_1 = var(x_1);
emp_var_2 = var(x_2);
emp_var_sum = var(x_sum);

%%
emp_sum = 0.5 * (x_1 + 0.2);
emp_mean_emp_sum = mean(emp_sum);
emp_var_emp_sum = var(emp_sum);