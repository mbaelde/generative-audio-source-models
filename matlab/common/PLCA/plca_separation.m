function [P_s,P_t] = plca_separation(V,Z,iter_max,P_f)

cum_Z = [0,cumsum(Z)];
n_source = length(Z);
[F,T] = size(V);

P_s = rand(n_source,T);
%P_s = P_s ./ repmat(sum(P_s,1),[n_source,1]);

P_t = rand(cum_Z(end),T);
%P_t = P_t ./ repmat(sum(P_t,2),[1,1]);

%rV = repmat(reshape(V,[1,1,F,T]),[n_source,Z(1),1,1]);
rV = repmat(reshape(V,[1,F,T]),[cum_Z(end),1,1]);
for iter = 1:iter_max
    % E-Step
    %P_szft = zeros(n_source,Z(1),F,T);
    P_szft = zeros(cum_Z(end),F,T);
    norm_coeff = 0;
    for s = 1:n_source
        aux = repmat(reshape(P_t(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),1,T]),[1,F,1]) .* repmat(reshape(P_f(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),F,1]),[1,1,T]);
        P_szft(cum_Z(s)+1:cum_Z(s+1),:,:) = repmat(reshape(P_s(s,:),[1,1,T]),[Z(s),F,1]) .* aux;
        norm_coeff = norm_coeff + (repmat(reshape(P_s(s,:),[1,1,T]),[1,F,1]) .* sum(aux,1));
    end
    P_szft = P_szft ./ repmat(norm_coeff,[cum_Z(end),1,1]);
%     aux = repmat(reshape((P_s * ones(1,T)) .* P_t,[Z,1,T]),[1,F,1]);
%     P_zft = aux .* repmat(P_f,[1,1,T]);
%     P_zft = P_zft ./ repmat(sum(P_zft,1),[Z,1,1]);
    
    % M-step
    VP = rV .* P_szft;
    VP2 = reshape(sum(VP,2),[cum_Z(end),T]);
    VP32 = zeros(n_source,T);
    for s = 1:n_source
        VP32(s,:) = sum(VP2(cum_Z(s)+1:cum_Z(s+1),:),1);
    end
    
    P_s = VP32;
    P_s = P_s ./ repmat(sum(P_s,1),[n_source,1]);

    for s = 1:n_source
        P_t(cum_Z(s)+1:cum_Z(s+1),:) = VP2(cum_Z(s)+1:cum_Z(s+1),:) ./ repmat(VP32(s,:),[Z(s),1]);
    end
    %progressbar(iter,iter_max)
end
