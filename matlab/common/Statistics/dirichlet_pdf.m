function density = dirichlet_pdf(r,a)

density = (gammaln(sum(a)) - sum(gammaln(a))) + sum(repmat((a-1),[size(r,1),1]) .* log(r),2);


