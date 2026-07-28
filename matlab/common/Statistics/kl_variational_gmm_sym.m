function dist = kl_variational_gmm_sym(p_1, mu_1, sigma_1, p_2, mu_2, sigma_2)

dist = kl_variational_gmm_mex(p_1, mu_1, sigma_1, p_2, mu_2, sigma_2) + kl_variational_gmm_mex(p_2, mu_2, sigma_2, p_1, mu_1, sigma_1);