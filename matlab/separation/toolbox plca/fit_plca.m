function dict = fit_plca(spectrogram, z, iter_max)

[F,T] = size(spectrogram);

dict.freq_basis = rand(F,length(z));
dict.temp_act = rand(T,length(z));
posterior = zeros(F,length(z),T);
phi = zeros(F,length(z));
phi_t = zeros(T,length(z));

for iter_cur = 1:iter_max
    for t = 1:T
        % E-step
        posterior(:,:,t) = repmat(dict.temp_act(t,:),[F,1]) .* dict.freq_basis;
        posterior(:,:,t) = posterior(:,:,t) ./ repmat(sum(posterior(:,:,t),2),[1,length(z)]);
    end
    
    % M-step
    for zz = z
        phi(:,zz) = sum(reshape(posterior(:,zz,:),[F,T]) .* spectrogram,2);
        phi_t(:,zz) = sum(reshape(posterior(:,zz,:),[F,T]) .* spectrogram,1);
    end
    dict.freq_basis = phi ./ repmat(sum(phi,1),[F,1]);
    dict.temp_act = phi_t ./ repmat(sum(phi_t,2),[1,length(z)]);
    progressbar(round(iter_cur),iter_max)
end