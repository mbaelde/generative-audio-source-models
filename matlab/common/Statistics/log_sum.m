function output = log_sum(input)
% input is assumed sorted in the descend order : input(1) > input(2) >...

if length(input) == 2
    output = log(1 + exp(input(2) - input(1)));
else
    output = log(1 + exp(input(2) - input(1) + log_sum(input(2:end))));
end

output = input(1) + output;