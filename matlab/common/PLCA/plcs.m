function [P_z,P_f,P_t] = plcs(V,Z,iter_max,fixed)

[F,T,I] = size(V);

if isempty(fixed{1})
    P_z = rand(Z,I);
    P_z = P_z ./ repmat(sum(P_z),[Z,1]);
else
    P_z = fixed{1};
end
if isempty(fixed{2})
    P_f = rand(Z,F);
    P_f = P_f ./ repmat(sum(P_f,2),[1,F]);
else
    P_f = fixed{2};
end
if isempty(fixed{3})
    P_t = rand(Z,T);
    P_t = P_t ./ repmat(sum(P_t,2),[1,T]);
else
    P_t = fixed{3};
end
rV = repmat(reshape(V,[1,F,T,I]),[Z,1,1,1]);
for iter = 1:iter_max
%     figure(1)
%     clf
%     plot(db(P_f)')
%     pause(0.001)
    % E-Step
    for ch = 1:I
        aux(:,:,:,ch) = repmat(reshape( (P_z(:,ch) * ones(1,T)) .* P_t,[Z,1,T]),[1,F,1]);
    end
    P_czft = aux .* repmat(P_f,[1,1,T,I]);
    P_czft = P_czft ./ repmat(sum(P_czft,1),[Z,1,1,1]);
    
    % M-step
    VP = rV .* P_czft;
    VPch = sum(VP,4);
    VP2 = reshape(sum(VPch,2),[Z,T]);
    VP3 = sum(VPch,3);
    VP32 = sum(VP3,2);
    if isempty(fixed{1})
        P_z = reshape(sum(sum(VP,2),3),[Z,I]);
        P_z = P_z ./ repmat(sum(P_z,1),[Z,1]);
    end
    if isempty(fixed{2})
        P_f = VP3 ./ repmat(VP32,[1,F]);
    end
    if isempty(fixed{3})
        P_t = VP2 ./ repmat(VP32,[1,T]);
    end
    progressbar(iter,iter_max)
end
