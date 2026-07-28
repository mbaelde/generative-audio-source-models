function [P_s,P_t] = plcs_separation(V,Z,iter_max,P_f)

cum_Z = [0,cumsum(Z)];
n_source = length(Z);
[F,R,I] = size(V);

P_s = rand(n_source,R,I);

P_t = rand(cum_Z(end),R);

rV = repmat(reshape(V,[1,F,R,I]),[cum_Z(end),1,1,1]);
for iter = 1:iter_max
    % E-Step
    P_szft = zeros(cum_Z(end),F,R,I);
    norm_coeff = zeros(F,R,I);
    for s = 1:n_source
        aux = repmat(reshape(P_t(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),1,R]),[1,F,1]) .* repmat(reshape(P_f(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),F,1]),[1,1,R]);
        for i = 1:I
            P_szft(cum_Z(s)+1:cum_Z(s+1),:,:,i) = repmat(reshape(P_s(s,:,i),[1,1,R]),[Z(s),F,1]) .* aux;
%             if R == 1
%                 norm_coeff(:,:,i) = norm_coeff(:,:,i) +(repmat(reshape(P_s(s,:,i),[1,1,R]),[1,F,1]) .* sum(aux,1))';
%             else
                norm_coeff(:,:,i) = norm_coeff(:,:,i) + permute(repmat(reshape(P_s(s,:,i),[1,1,R]),[1,F,1]) .* sum(aux,1),[2,3,1]);
%             end
        end
        %norm_coeff = norm_coeff + (repmat(reshape(P_s(s,:,i),[1,1,R]),[1,F,1]) .* sum(aux,1));
    end
    P_szft = P_szft ./ repmat(reshape(norm_coeff,[1,F,R,I]),[cum_Z(end),1,1,1]);
    
    % M-step
    VP = rV .* P_szft;
    VP2 = reshape(sum(VP,2),[cum_Z(end),R,I]);
    VP2ch = sum(VP2,3);
    VP32 = zeros(n_source,R,I);
    for s = 1:n_source
        VP32(s,:,:) = sum(VP2(cum_Z(s)+1:cum_Z(s+1),:,:),1);
    end
    VP32ch = sum(VP32,3);
    
    P_s = VP32;
    P_s = P_s ./ repmat(sum(P_s,1),[n_source,1,1]);

    for s = 1:n_source
        P_t(cum_Z(s)+1:cum_Z(s+1),:) = VP2ch(cum_Z(s)+1:cum_Z(s+1),:) ./ repmat(VP32ch(s,:),[Z(s),1]);
    end
    %progressbar(iter,iter_max)
end
