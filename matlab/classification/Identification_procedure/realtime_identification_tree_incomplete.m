function L_bay = realtime_identification_tree_incomplete(data, tree, subtree, param)

N_spect = param.N_spect;

n_buff = 10;
    
aux_decision = zeros(1,n_buff);

% prior_g = zeros(1,n_class);
% for k = 1:n_class
%     prior_g(k) = sum(database(:,end-1) ==k);
% end

cnt = 0;
for b = 1:n_buffer
    cnt = cnt + 1;
    % Calcul du spectre du buffer
    data = database(b,1:end-2);
    % Compute spectrum
    spectrum = abs(fft(data)).^2;
    spectrum_norm = N_spect * spectrum(1:N_spect) ./ sum(spectrum(1:N_spect));

    aux_decision(cnt) = browse_tree_incomplete(tree, subtree, spectrum_norm);
    
    if mod(cnt,n_buff) == 0
        cnt = 0;
        L_bay((b-n_buff+1):b) = mode(aux_decision);
    end
    progressbar(b,n_buffer)
end