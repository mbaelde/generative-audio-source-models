function density = pearsonpdf(x, lambda, a, m, nu, normalize)

k = (gamma(m) / (sqrt(pi) * a * gamma(m-1/2))) * abs( cgamma( m+ (1i * nu / 2)) / gamma(m) )^2;

if strcmp(normalize,'sum')
    density = (1 + ((x-lambda)/a).^2).^(-m) .* exp( - nu * atan((x-lambda)/a));
    density = density / sum(density);
elseif strcmp(normalize, 'k')
    density = k*(1 + ((x-lambda)/a).^2).^(-m) .* exp( - nu * atan((x-lambda)/a));
end
