function feature = compute_feature(spectrum,param)

N = size(spectrum,1);
n_fft = param.n_fft;
power = param.power;
% q = param.q;

% feature = abs(spectrum(:,1:n_fft)).^power;
% feature = feature ./ repmat(sum(feature,2),[1,n_fft]);
% feature = [feature,spectrum(:,end-1:end)];

feature = zeros(N,n_fft+2);
for n = 1:N
    aux = abs(spectrum(n,1:n_fft)).^power;
    aux = aux ./ sum(aux);
%     tilde_x = floor(q * aux);
%     tilde_q = sum(tilde_x);
%     aux = tilde_x / tilde_q;
    feature(n,:) = [aux, spectrum(n,end-1:end)];
end
% feature(feature == 0) = eps;
