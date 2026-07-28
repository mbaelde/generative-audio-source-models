function out_temp = fit_plca_supervised(V, z, input_dict, iter_max)

F = size(V,1);

N_z = [length(z{1}), length(z{2})];

posterior = zeros(F,sum(N_z));
out_temp = rand(1,sum(N_z));
out_temp(1:N_z(1)) = out_temp(1:N_z(1)) ./ sum(out_temp(1:N_z(1)));
out_temp(N_z(1)+1:sum(N_z)) = out_temp(N_z(1)+1:sum(N_z)) ./ sum(out_temp(N_z(1)+1:sum(N_z)));

for iter_cur = 1:iter_max
    % E-step
    posterior(:,1:N_z(1)) = repmat(out_temp(1:N_z(1)),[F,1]) .* input_dict{1}.freq_basis;
    posterior(:,N_z(1)+1:sum(N_z)) = repmat(out_temp(N_z(1)+1:sum(N_z)),[F,1]) .* input_dict{2}.freq_basis;
    posterior = posterior ./ repmat(sum(posterior,2),[1,sum(N_z)]);
    
    % M-step
    phi_t = sum(posterior .* repmat(V,[1,sum(N_z)]),1);
    out_temp(1:N_z(1)) = phi_t(1:N_z(1)) ./ repmat(sum(phi_t(1:N_z(1)),2),[1,N_z(1)]);
    out_temp(N_z(1)+1:sum(N_z)) = phi_t(1+N_z(1):sum(N_z)) ./ repmat(sum(phi_t(1+N_z(1):sum(N_z)),2),[1,N_z(2)]);
end