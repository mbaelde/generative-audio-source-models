function out = LSE_normprob(x)

[~,M] = size(x);
norm_factor = LSE(x);
out = x - repmat(norm_factor,[1,M]);