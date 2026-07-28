function fx = min_alpha(alpha_ip, salpha, xn, zi)

fx = psi(alpha_ip) - psi(salpha) - sum(zi .* log(xn)) ./ sum(zi);