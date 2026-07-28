[audio,fs] = audioread('Polyphonic sounds/audio001.wav');

N_window = 1024;
Window = hanning(N_window);
Nshift = N_window / 2;
Nfft = N_window;
spectrogram_complex = spectrogram(audio,sqrt(Window),Nshift,Nfft);

V = abs(spectrogram_complex);
[F,N] = size(V);

K = 5;

alpha = ones(1,K);
beta = ones(1,K);

%% Initialize parameters
W = abs(randn(F,K)) + ones(F,K);
H = abs(randn(K,N)) + ones(K,N);

% MJLE
iter_max = 100;
for iter = 1:iter_max
    W_old = W;
    H_old = H;
    % Update H
    WH_old = W_old*H_old;
    for k = 1:K
        aux = sum(repmat(W(:,k),[1,N]) .* V ./ WH_old,1);
        H(k,:) = H_old(k,:) .* (aux + alpha(k) - 1) ./ ((1+1/beta(k))*sum(W(:,k)));
    end
    % Update W
    W_oldH = W_old*H_old;
    for k = 1:K
        aux = sum(repmat(H(k,:),[F,1]) .* V ./ W_oldH,2);
        W(:,k) = W_old(:,k) .* (aux + (alpha(k) - 1)*N / sum(abs(W(:,k)))) ./ ((1+1/beta(k))*sum(H(k,:)));
    end
    progressbar(iter,iter_max)
end

V_hat = W*H;

figure(1)
clf
imagesc(db(V))

figure(2)
clf
imagesc(db(V_hat))

%% Initialize parameters
W = abs(randn(F,K)) + ones(F,K);
H = abs(randn(K,N)) + ones(K,N);

alpha_b = ones(K,N);
beta_b = ones(K,1);
p = zeros(K,F,N);

% MJLE
iter_max = 100;
for iter = 1:iter_max
    W_old = W;
    H_old = H;
    % E-step
    log_H = psi(alpha_b) + repmat(log(beta_b),[1,N]);
    for k = 1:K
        p(k,:,:) = repmat(W_old(:,k),[1,N]) .* repmat(exp(log_H(k,:)),[F,1]);
    end
    p = p ./ repmat(sum(p,1),[K,1,1]);
    
    C = p .* repmat(reshape(V,[1,F,N]),[K,1,1]);
    alpha_b = repmat(alpha',[1,N]) + reshape(sum(C,2),[K,N]);
    beta_b = 1./ (1./beta + sum(W_old,1))';
    
    H = alpha_b .* repmat(beta_b,[1,N]);
    
    % M-step
    W = sum(C,3)' ./ repmat(sum(H,2)',[F,1]);
    
    progressbar(iter,iter_max)
end

V_hat = W*H;

figure(1)
clf
imagesc(db(V))

figure(2)
clf
imagesc(db(V_hat))