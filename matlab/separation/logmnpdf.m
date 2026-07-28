function out = logmnpdf(x,p)
q = sum(x);

out = gammaln(q+1) - sum(gammaln(x+1)) + sum(x .* log(p));