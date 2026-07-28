function out = mixture_dirichletpdf(x, a, weight)

M = length(weight);
N = size(x,1);

out = zeros(M,N);
for m = 1:M
    out(m,:) = log(weight(m)) + dirichlet_pdf(x, a(m,:));
end
