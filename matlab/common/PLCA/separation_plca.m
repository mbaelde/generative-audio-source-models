function [P_s,P_t] = separation_plca(V,P_f,iter_max)

[F,T] = size(V);

S = length(P_f);
Z = zeros(1,S);
P_t = cell(1,S);
for s = 1:S
    Z(s) = size(P_f{s},1);
    P_t{s} = rand(Z(s),1,T);
    P_t{s} = P_t{s} ./ repmat(sum(P_t{s},1),[Z(s),1,1]);
end
P_s = rand(2,1,T);
P_s = P_s ./ repmat(sum(P_s),[S,1,1]);

rV = reshape(V,[1,F,T]);

P_exp = cell(1,S);
for iter = 1:iter_max
    % E-Step   
    norm_factor = zeros(1,F,T);
    for s = 1:S
        P_exp{s} = repmat(P_s(s,:,:),[Z(s),F,1]) .* repmat(P_t{s},[1,F,1]) .* repmat(P_f{s},[1,1,T]);
        norm_factor = norm_factor + sum(P_exp{s},1);
    end
    for s = 1:S
        P_exp{s} = P_exp{s} ./ repmat(norm_factor,[Z(s),1,1]);
    end
    
    % M-step
    for s = 1:S
        aux = P_exp{s} .* repmat(rV,[Z(s),1,1]);
        sum_aux = sum(aux,2);
        P_s(s,:,:) = sum(sum_aux,1);
        P_t{s} = sum_aux;
    end
    P_s = P_s ./ repmat(sum(P_s),[S,1,1]);
    for s = 1:S
        norm_z = sum(P_t{s},1);
        P_t{s} = P_t{s} ./ repmat(norm_z,[Z(s),1,1]);
    end
    progressbar(iter,iter_max)
end
