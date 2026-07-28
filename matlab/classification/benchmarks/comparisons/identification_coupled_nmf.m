function [activation_matrix_test,computation_time] = identification_coupled_nmf(database_test, W, param)

n_class = param.n_class;
F = size(W,1) - n_class;
K = size(W,2);
N = size(database_test,1);
computation_time = zeros(1,N);
activation_matrix_test = zeros(n_class,N);
R = 10;
iter_max = 100;
W_old = W(1:F,:);

for b = 1:N
    if mod(b,R) == 0
        tic
        data = database_test(b-R+1:b,1:end-2);

        spectrum = abs(fft(data,[],2));
        V_test = spectrum(:,1:param.n_fft);
    
        H = abs(randn(K,R)) + ones(K,R);
        alpha = ones(K,1);
        beta = ones(K,1);
        alpha_b = ones(K,R);
        beta_b = ones(K,1);
        p = zeros(K,F,R);
        
        for iter = 1:iter_max
            %H_old = H;
            % E-step
            log_H = psi(alpha_b) + repmat(log(beta_b),[1,R]);
            for k = 1:K
                p(k,:,:) = repmat(W_old(:,k),[1,R]) .* repmat(exp(log_H(k,:)),[F,1]);
            end
            p = p ./ repmat(sum(p,1),[K,1,1]);

            C = p .* repmat(reshape(V_test,[1,F,R]),[K,1,1]);
            alpha_b = repmat(alpha,[1,R]) + reshape(sum(C,2),[K,R]);
            beta_b = 1./ (1./beta + sum(W_old,1)');

            H = alpha_b .* repmat(beta_b,[1,R]);
            
%             aux_act = W(F+1:end,:) * H;
%             figure(1)
%             clf
%             imagesc(aux_act)
%             figure(2)
%             clf
%             plot(H')
%             pause(0.01)
            
        end
        
        activation_matrix_test(:,b-R+1:b) = W(F+1:end,:) * H;

        computation_time(b) = toc;
    end
    progressbar(b,N)
end
