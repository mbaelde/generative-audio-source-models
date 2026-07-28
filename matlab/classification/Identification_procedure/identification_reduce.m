function L_bay = identification_reduce(database, dictionary, my_class, param)
% Set parameters
N_spect = param.N_spect;
winsize = param.winsize;
fs = param.fs;
type = param.type;

% Precompute things

if strcmp(type,'clavel')
    n_buff = floor((0.5 * fs) / winsize);
else
    n_buff = 1;
end

n_class = length(my_class);


N = size(database,1);
dict_size = size(dictionary,1);
% Initialize variables
L_bay = zeros(1,N);
L = zeros(n_buff, dict_size);

cnt = 0;
% Identification
for b = 1:N
    cnt = cnt + 1;
    % Calcul du spectre du buffer
    data = database(b,1:end-2)';
    
    % Compute spectrum
    spectrum = abs(fft(data)).^2;
    spectrum_norm = N_spect * spectrum(1:N_spect) ./ sum(spectrum(1:N_spect));
    
    % Calcul de la vraisemblance
    rspectrum_norm = repmat(spectrum_norm, [1,n_class])';
    L(cnt,:) = sum(rspectrum_norm .* dictionary,2);    
       
    % Cherche la classe correspondance
    if mod(b, n_buff) == 0
        sum_L = sum(L,1);
        cnt = 0;
        
        % Method 2 : Full Bayesian
        L_bay((b-n_buff+1):b) = my_class(max(sum_L) == sum_L);
    end
    
    progressbar(b,N)
end

