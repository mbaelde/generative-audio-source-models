function [segments, approx] = discretize_univariate_normpdf(n_segments, model, th)

mu = model.mu;
sigma = model.sigma;
if isfield(model,'p')
    p = model.p;
end

segments = zeros(n_segments,1);
approx = zeros(n_segments-1,1);
syms solx 
if isfield(model,'p')
    segments(1) = vpasolve(mixture_normcdf(solx,mu,sigma,p) == th,solx);
    for k = 2:n_segments
        segments(k) = vpasolve(mixture_normcdf(solx,mu,sigma,p) - mixture_normcdf(segments(k-1),mu,sigma,p) == 1/n_segments,solx);
    end
    segments(end) = vpasolve(mixture_normcdf(solx,mu,sigma,p) == 1-th,solx);
    for k = 1:n_segments-1
        approx(k) = mixture_normpdf(mean([segments(k),segments(k+1)]),mu,sigma,p);
    end
else
    segments(1) = vpasolve(normcdf(solx,mu,sigma) == th,solx);
    for k = 2:n_segments
        segments(k) = vpasolve(normcdf(solx,mu,sigma) - normcdf(segments(k-1),mu,sigma) == 1/n_segments,solx);
    end
    segments(end) = vpasolve(normcdf(solx,mu,sigma) == 1-th,solx);
    for k = 1:n_segments-1
        approx(k) = normpdf(mean([segments(k),segments(k+1)]),mu,sigma);
    end
end