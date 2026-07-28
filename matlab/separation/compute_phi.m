function prob_phi = compute_phi(x_1, y, p_1, p_2)

M_1 = size(p_1,2);
M_2 = size(p_2,2);

q = length(y);

denom = zeros(M_1,M_2);
for j = 1:M_1
    for k = 1:M_2
        p = 0.5 * (p_1(:,j) + p_2(:,k));
        denom(j,k) = logmnpdf(y, p); 
    end
end

prop_norm = zeros(1,M_2);
for k = 1:M_2
    prop_norm(k) = logmnpdf(y-x_1, p_2(:,k));
end

prob_phi = LSE(prop_norm) - LSE(denom(:)');

