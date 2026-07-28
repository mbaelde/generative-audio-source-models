function [W,H] = learn_coupled_nmf(feature_training,activation_matrix_training,param)
%% Apply NMF with MMLE
% Initialize parameters
n_class = size(activation_matrix_training,1);
V_train = [feature_training(:,1:end-2)'; activation_matrix_training];

E = n_class;
[F,N] = size(V_train);
F = F - E;
K = param.K;

m = 20; %0.5sec long
n_bloc = round((N - m) / m);
W = abs(randn(F+E,K*n_bloc)) + ones(F+E,K*n_bloc);
H = abs(randn(K*n_bloc,m)) + ones(K*n_bloc,m);
iter_max = param.iter_max;
for nn = 1:n_bloc
    alpha = ones(K,1);
    beta = ones(K,1);

    alpha_b = ones(K,m);
    beta_b = ones(K,1);
    
    for iter = 1:iter_max
        W_old = W(:,(nn-1)*K+1:nn*K);
        % E-step
        log_H = psi(alpha_b) + repmat(log(beta_b),[1,m]);
        p = zeros(K,F+E,m);
        for k = 1:K
            p(k,:,:) = repmat(W_old(:,k),[1,m]) .* repmat(exp(log_H(k,:)),[F+E,1]);
        end
        norm_factor = sum(p,1);
        for k = 1:K
            p(k,:,:) = p(k,:,:) ./ norm_factor;
        end
        p(isnan(p)) = 0;
        for k = 1:K
            C = reshape(p(k,:,:),[F+E,m]) .* V_train(:,(nn-1)*m+1:nn*m);
            alpha_b(k,:) = alpha(k) + sum(C,1);
        end

    %     for k = 1:K
    %         C(k,:,:) = p(k,:,:) .* reshape(V_train,[1,F+E,N]);
    %     end
    %     
    %     alpha_b = repmat(alpha,[1,N]) + reshape(sum(C,2),[K,N]);
        beta_b = 1./ (1./beta + sum(W_old,1)');

        H((nn-1)*K+1:nn*K,:) = alpha_b .* repmat(beta_b,[1,m]);

        % M-step
        for k = 1:K
            C = reshape(p(k,:,:),[F+E,m]) .* V_train(:,(nn-1)*m+1:nn*m);
            W(:,(nn-1)*K+k) = sum(C,2) ./ sum(H(k,:),2);
        end
    %     W = sum(C,3)' ./ repmat(sum(H,2)',[F+E,1]);

        %progressbar(iter,iter_max)
%         figure(1)
%         clf
%         plot(W_old)
%         pause(0.01)
    end
    %progressbar(nn,n_bloc)
    clc
    disp(['nn: ',num2str(nn),' / ',num2str(n_bloc)])
end