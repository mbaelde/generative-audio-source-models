function [P_z,P_f,P_t] = plca_semisupervised(V,Z,iter_max,fixed)

[F,T] = size(V);

if isempty(fixed{1})
    P_z = rand(Z,1);
    P_z = P_z ./ sum(P_z);
else
    P_z = fixed{1};
end
if isempty(fixed{2})
    P_f = rand(Z,F);
    P_f = P_f ./ repmat(sum(P_f,1),[Z,1]);
else
    P_fknown = fixed{2};
    F_k = size(P_fknown,1);
    P_f = rand(Z,F);
    P_f = P_f ./ repmat(sum(P_f,1),[Z,1]);
end
if isempty(fixed{3})
    P_t = rand(Z,T);
    P_t = P_t ./ repmat(sum(P_t,1),[Z,1]);
else
    P_t = fixed{3};
end
rV = repmat(reshape(V,[1,F,T]),[Z,1,1]);

for iter = 1:iter_max
    % E-Step
    aux = repmat(reshape((P_z * ones(1,T)) .* P_t,[Z,1,T]),[1,F,1]);
    P_zft = aux .* repmat(P_f,[1,1,T]);
    P_zft = P_zft ./ repmat(sum(P_zft,1),[Z,1,1]);
    
    % M-step
    VP = rV .* P_zft;
    VP3 = sum(VP,3);
    if isempty(fixed{1})
        P_z = sum(VP3,2);
        P_z = P_z ./ sum(P_z);
    end
    if isempty(fixed{2})
        P_f = VP3 ./ repmat(P_z,[1,F]);
    end
    if isempty(fixed{3})
        P_t = reshape(sum(VP,2),[Z,T]) ./ repmat(P_z,[1,T]);
    end
    %progressbar(iter,iter_max)
end
