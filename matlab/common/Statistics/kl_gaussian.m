function dist = kl_gaussian(mu_1, sigma_1, mu_2, sigma_2)

dist = 0.5 * (log(sigma_2 / sigma_1) + ((sigma_1 + (mu_1 - mu_2)^2) / sigma_2) - 1);