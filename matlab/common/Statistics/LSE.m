function out = LSE(x)

[~,M] = size(x);

x_max = max(x,[],2);
out = x_max + log(sum(exp(x - repmat(x_max,[1,M])),2,'omitnan'));