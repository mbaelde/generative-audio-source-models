function [out_dict, out_temp] = fit_plca_learn(V, z, dict, B, alpha, previous_temp_act, iter_max)

F = size(V,1);
L = size(B,2);

N_z = [length(z{1}), length(z{2})];

posterior = zeros(F, sum(N_z), L);
out_temp = rand(1,length(z));

rB = repmat(reshape(B,[F,1,L]),[1,N_z(2),1]);
for iter_cur = 1:iter_max
    for s = 1:L
        % E-step
        posterior(:,1:N_z(1),s) = repmat(previous_temp_act(s,:),[F,1]) .* dict{1}.freq_basis;
        posterior(:,N_z(1)+1:sum(N_z),s) = repmat(dict{2}.temp_act(s,:),[F,1]) .* dict{2}.freq_basis;
        
        posterior(:,:,s) = posterior(:,:,s) ./ repmat(sum(posterior(:,:,s),2),[1,sum(N_z)]);
    end
    
    % M-step
    phi = repmat(V,[1,N_z(2)]) .* posterior(:,N_z(1)+1:sum(N_z),end) + (alpha / L) * sum(posterior(:,N_z(1)+1:sum(N_z),:) .* rB,3);
    phi_t = sum(posterior(:,:,end) .* repmat(V,[1,sum(N_z)]),1);
    
    out_dict.freq_basis = phi ./ repmat(sum(phi,1),[F,1]);
    out_dict.temp_act = phi_t(1+N_z(1):sum(N_z)) ./ repmat(sum(phi_t(1+N_z(1):sum(N_z)),2),[1,N_z(2)]);
    out_temp = phi_t(1:N_z(1)) ./ repmat(sum(phi_t(1:N_z(1)),2),[1,N_z(1)]); % for source 1

end