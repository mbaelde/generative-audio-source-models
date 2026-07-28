function [mask, likelihood_y] = compute_masking(y, p_1, p_2)

M_1 = size(p_1,2);
M_2 = size(p_2,2);
F = length(y);
q = F;

aux = zeros(M_1,M_2);
ratio_p_1 = zeros(M_1,M_2,F);
ratio_p_2 = zeros(M_1,M_2,F);
for j = 1:M_1
    for k = 1:M_2
        aux(j,k) = logmnpdf(y, 0.5 * (p_1(:,j) + p_2(:,k)));
        ratio_p_1(j,k,:) = log(p_1(:,j)) - log(p_1(:,j) + p_2(:,k));
        ratio_p_2(j,k,:) = log(p_2(:,k)) - log(p_1(:,j) + p_2(:,k));
    end
end
likelihood_y = LSE(aux(:)');
phi_prob = 2 * q * log(2) + 2 * gammaln(q+1) - gammaln(2 * q + 1) + aux - likelihood_y;

hat_x_1 = zeros(F,1);
hat_x_2 = zeros(F,1);
for f = 1:F
    sum_aux_1 = phi_prob + ratio_p_1(:,:,f);
    sum_aux_2 = phi_prob + ratio_p_2(:,:,f);
    hat_x_1(f) = exp( log(y(f)) + LSE(sum_aux_1(:)'));
    hat_x_2(f) = exp( log(y(f)) + LSE(sum_aux_2(:)'));
end

mask = hat_x_1 ./ (hat_x_1 + hat_x_2);