% Generate data
p = 50;
m = 25;

l = 21;
n = 50000;

omega = randn(p,m);
omega = (omega - repmat(mean(omega,2),[1,m])) ./ repmat(std(omega,[],2),[1,m]);

Y = randn(m,n);
Y = (Y - repmat(mean(Y,2),[1,n])) ./ repmat(std(Y,[],2),[1,n]);

YYp = Y*Y';

% apply to audio data
[sound,fs] = audioread('rare_program/Reduced dictionary based/test_sounds/damien.mp3');
window = hanning(1024);
nshift = 512;
nfft = 1024;
Y = abs(spectrogram(sound,window,nshift,nfft));

% Algorithm Analysis SimCO
iter_max = 100;
t = 1e-3;
for iter = 1:iter_max
    % update X
    X = omega * Y;
    for nn = 1:n
        data = X(:,nn);
        [~,idx] = sort(data);
        X(idx(1:l),nn) = 0;
    end
    % update omega
    H = 2*X*Y' - 2*omega * YYp; 
    h_bar = zeros(size(H));
    omega_t = zeros(size(omega));
    for j = 1:p
        h_bar(j,:) = H(j,:) - H(j,:) * omega(j,:)' * omega(j,:);
        norm_h = sum(h_bar(j,:).^2);
        if norm_h == 0
            omega_t(j,:) = omega(j,:);
        else
            omega_t(j,:) = omega(j,:) * cos(norm_h * t) + (h_bar(j,:) ./ norm_h) * sin(norm_h * t);
        end
    end
    omega = omega_t;
    progressbar(iter,iter_max)
end