%% Implementation of Duan's technique: Only PLCA
% From: "Online PLCA for Real-Time Semi-Supervised Source Separation",
% Duan et al, International Conference on Latent Variable Analysis and
% Signal Separation, 2012

clear
clc
addpath(genpath('Statistics'))
addpath(genpath('toolbox plca'))
%% Load data
[source_1,~] = audioread('dog_1.wav');
source(1,:) = source_1;
[source_2,~] = audioread('laugh_1.wav');
source(2,:) = source_2;
[mixture,fs] = audioread('mixture_dog_laugh.wav');
clear source_1 source_2

%%
[mixture,fs] = audioread('Data/Denoising/mixture.wav');
mixture = mixture ./ max(abs(mixture));

source_1 = repmat(mixture(1:10000),[5,1]);
source_1 = source_1(1:length(mixture));

source = [source_1, mixture]';

%%
n_window = 1024;
n_shift = 512;
Window = hamming(n_window);
%% Compute a dictionary for source 1 (Shashanka's thesis, page 45)
for s = 1:2
    aux = abs(spectrogram(source(s,:),Window,n_shift,n_window));
    [F,T] = size(aux);
    spectrogram_s(:,:,s) = aux;
    energy(s,:) = sum(spectrogram_s(:,:,s),1);
    spectrogram_s(:,:,s) = spectrogram_s(:,:,s) ./ repmat(energy(s,:),[F,1]);
end
clear aux

z{1} = 1:100;
iter_max = 100;

% Fit dictionary
dict{1} = fit_plca(spectrogram_s(:,:,1), z{1}, iter_max);

%% Learn theta_kl threshold
% Reconstruct the sound based on this dictionary
aux_spect = dict{1}.freq_basis * dict{1}.temp_act';

d_kl = zeros(1,T);
for t = 1:T
    d_kl(t) = KLDiv(spectrogram_s(:,t,1)', aux_spect(:,t)');
end
theta_kl = mean(d_kl) + std(d_kl);

%% Semi-supervised separation
% Mixture spectrogram
V = abs(spectrogram(mixture,Window,n_shift,n_window));
F = size(V,1); % Number of frequency bins
T = size(V,2); % Number of time frames
energy_v = sum(V,1);
V = V ./ repmat(energy_v,[F,1]);
% Running buffers
L = 60; % Number of buffers to keep
B = zeros(F,L);
% Source 2 dictionary
z{2} = 1:20;
dict{2}.freq_basis = rand(F,length(z{2}));
dict{2}.temp_act = rand(T,length(z{2}));
% Temporal activation of the mixture: used to decide is S2 is in the
% mixture
%mixture_act = rand(1,length(z{1}));
% EM parameters
iter_max = 20;
alpha = 20;
% Reconstructed spectrograms
rec_spectrogram = zeros(F,T,2);
temp_act = rand(T,length(z{1}));
previous_temp_act = zeros(L,length(z{1}));
previous_temp_act(end,:) = temp_act(1,:);

for t = 1:T
    % Decompose mixture spectrogram
    mixture_act = fit_plca_mixture(V(:,t), z{1}, dict{1}, iter_max);
    % Compute the approximation based on the previous decomposition
    approx = sum(dict{1}.freq_basis .* repmat(mixture_act,[F,1]),2);
    d_kl_mixture = KLDiv(V(:,t)', approx');
    
    if d_kl_mixture < theta_kl
        % Supervised separation
        input_dict = dict;
        input_dict{2}.temp_act = dict{2}.temp_act(t,:);
        [out_dict,out_temp] = fit_plca_supervised(V(:,t), z, input_dict, iter_max);
        dict{2}.temp_act(t,:) = out_dict.temp_act;
        temp_act(t,:) = out_temp;
    else
        % Fit dictionary for the source 2
        [out_dict, out_temp] = fit_plca_learn(V(:,t), z, dict, B, alpha, previous_temp_act, iter_max);
        dict{2}.freq_basis = out_dict.freq_basis;
        dict{2}.temp_act(t,:) = out_dict.temp_act;
        temp_act(t,:) = out_temp;
        
        previous_temp_act = circshift(previous_temp_act,-1,1);
        previous_temp_act(end,:) = temp_act(t,:);
        % Update B
        B = circshift(B,-1,2);
        B(:,end) = V(:,t);
    end
    % Reconstruct sources
    num(:,1) = sum(dict{1}.freq_basis .* repmat(temp_act(t,:),[F,1]),2);
    num(:,2) = sum(dict{2}.freq_basis .* repmat(dict{2}.temp_act(t,:),[F,1]),2);
    denom = sum(num,2);
    rec_spectrogram(:,t,1) = V(:,t) .* num(:,1) ./ denom;
    rec_spectrogram(:,t,2) = V(:,t) .* num(:,2) ./ denom;
    progressbar(t,T)
end

% Reconstruct the sources and store the results
for s = 1:2
    spectrogram_dict = rec_spectrogram(:,:,s) .* repmat(energy(s,:),[F,1]);

    S = spectrogram_dict;
    out(1:n_shift*(T-1)+n_window,1)=0;

    for i=1:T
        Sfft=[S(:,i);conj(S(n_window/2:-1:2,i))];
        stemp=real(ifft(Sfft));
        sout=stemp(1:n_window,1) .* Window;

        out(n_shift*(i-1)+1:n_shift*(i-1)+n_window,1)=out(n_shift*(i-1)+1:n_shift*(i-1)+n_window,1)+sout;
    end

    audiowrite(['Results/Denoising/source_',num2str(s),'_reconstruct.wav'],out,fs)
end
