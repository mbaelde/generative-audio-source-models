function [P_s,P_f,P_t] = plcs_semiseparation(V,Z,iter_max,P_f,id_upd)

cum_Z = [0,cumsum(Z)];
n_source = length(Z);
[F,R,I] = size(V);

P_s = rand(n_source,R,I);

% P_fn = P_f;
% P_fn(id_upd,:) = rand(Z(2),F);
P_t = rand(cum_Z(end),R);

rV = repmat(reshape(V,[1,F,R,I]),[cum_Z(end),1,1,1]);
for iter = 1:iter_max
%     figure(1)
%     clf
%     subplot(2,1,1)
%     plot(P_f(id_upd,:)')
%     subplot(2,1,2)
%     plot(P_f(1:50,:)')
%     figure(2)
%     clf
%     plot(P_t)
%     pause(0.0001)
    % E-Step
    P_szft = zeros(cum_Z(end),F,R,I);
    norm_coeff = zeros(F,R,I);
    for s = 1:n_source
        aux = repmat(reshape(P_t(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),1,R]),[1,F,1]) .* repmat(reshape(P_f(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),F,1]),[1,1,R]);
        for i = 1:I
            P_szft(cum_Z(s)+1:cum_Z(s+1),:,:,i) = repmat(reshape(P_s(s,:,i),[1,1,R]),[Z(s),F,1]) .* aux;
            norm_coeff(:,:,i) = norm_coeff(:,:,i) + reshape(repmat(reshape(P_s(s,:,i),[1,1,R]),[1,F,1]) .* sum(aux,1),[F,R]);
        end
    end
    P_szft = P_szft ./ repmat(reshape(norm_coeff,[1,F,R,I]),[cum_Z(end),1,1,1]);
    
    % M-step
    VP = rV .* P_szft;
    VP2 = reshape(sum(VP,2),[cum_Z(end),R,I]);
    VP2ch = sum(VP2,3);
    VPch = sum(VP,4);
    VP3 = sum(VPch,3);
    VP32 = zeros(n_source,R,I);
    for s = 1:n_source
        VP32(s,:,:) = sum(VP2(cum_Z(s)+1:cum_Z(s+1),:,:),1);
    end
    VP32ch = sum(VP32,3);
    
    P_s = VP32;
    P_s = P_s ./ repmat(sum(P_s,1),[n_source,1,1]);
    
    P_f(id_upd,:) = VP3(id_upd,:) ./ repmat(sum(VP3(id_upd,:),2),[1,F]);
    
    
    for s = 1:n_source
        P_t(cum_Z(s)+1:cum_Z(s+1),:) = VP2ch(cum_Z(s)+1:cum_Z(s+1),:) ./ repmat(VP32ch(s,:),[Z(s),1]);
    end
    %progressbar(iter,iter_max)
end
