function f = myfun(prop_em, model, mixture_pdf)
n_class = length(model);
for k = 1:n_class
    f(k) = prop_em(k) + LSE(mixture_dirichletpdf(mixture_pdf, model{k}.alpha, model{k}.prop)');
end
f = LSE(f);