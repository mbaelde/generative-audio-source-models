clear
clc
addpath(genpath('Toolbox'))
addpath(genpath('Prototypes/Separation'))
%% Load data
[mixture,fs] = audioread('..\Data\MUS2016\DSD100\Mixtures\Dev\052 - ANiMAL - Easy Tiger\mixture.wav');
source_1 = audioread('..\Data\MUS2016\DSD100\Sources\Dev\052 - ANiMAL - Easy Tiger\bass.wav');
source_2 = audioread('..\Data\MUS2016\DSD100\Sources\Dev\052 - ANiMAL - Easy Tiger\drums.wav');
source_3 = audioread('..\Data\MUS2016\DSD100\Sources\Dev\052 - ANiMAL - Easy Tiger\other.wav');
source_4 = audioread('..\Data\MUS2016\DSD100\Sources\Dev\052 - ANiMAL - Easy Tiger\vocals.wav');
% Transpose to get n_channel x n_samples
if size(source_1,1) > size(source_1,2)
    source_1 = source_1';
end
if size(source_2,1) > size(source_2,2)
    source_2 = source_2';
end
if size(source_3,1) > size(source_3,2)
    source_3 = source_3';
end
if size(source_4,1) > size(source_4,2)
    source_4 = source_4';
end
if size(mixture,1) > size(mixture,2)
    mixture = mixture';
end
% Mean to get mono channel
if size(source_1,1) > 1
    source_1 = mean(source_1);
end
if size(source_2,1) > 1
    source_2 = mean(source_2);
end
if size(source_3,1) > 1
    source_3 = mean(source_3);
end
if size(source_4,1) > 1
    source_4 = mean(source_4);
end
if size(mixture,1) > 1
    mixture = mean(mixture);
end

source = [source_1;source_2;source_3;source_4];
clear source_1 source_2 source_3 source_4
%% Compute spectrogram
N_window = 512;
Window = hanning(N_window);
Nshift = N_window / 2;
Nfft = N_window;
spectrogram_complex = spectrogram(mixture,sqrt(Window),Nshift,Nfft);

V = abs(spectrogram_complex);
phase_spect = angle(spectrogram_complex);

[F,T] = size(V);
figure(1)
imagesc(db(V))
%% Learn PLCA for the voice
spectro_voice = spectrogram(source(4,1027809:1882791),sqrt(Window),Nshift,Nfft);
V_voice = abs(spectro_voice);

z{1} = 1:20;
iter_max = 100;
% Fit dictionary
dict{1} = fit_plca(V_voice, z{1}, iter_max);
% save('Prototypes/Separation/Results/dict_voice_plca.mat','dict')
% load('Prototypes/Separation/Results/dict_voice_plca.mat')

%% Learn theta_kl threshold
% Reconstruct the sound based on this dictionary
aux_spect = dict{1}.freq_basis * dict{1}.temp_act';

T_voice = size(V_voice,2);
d_kl = zeros(1,T_voice);
for t = 1:T_voice
    d_kl(t) = KLDiv(V_voice(:,t,1)', aux_spect(:,t)');
end
theta_kl = mean(d_kl) + std(d_kl);

%% Semi-supervised separation
% Mixture spectrogram
energy_v = sum(V,1);
P_V = V ./ repmat(energy_v,[F,1]);
% Running buffers
L = 60; % Number of buffers to keep
B = zeros(F,L);
% Source 2 dictionary
z{2} = 1:20;
dict{2}.freq_basis = rand(F,length(z{2}));
dict{2}.temp_act = rand(T,length(z{2}));
% EM parameters
iter_max = 20;
alpha = 20;
% Reconstructed spectrograms
rec_spectrogram = zeros(F,T,2);
temp_act = rand(T,length(z{1}));
previous_temp_act = zeros(L,length(z{1}));
previous_temp_act(end,:) = temp_act(1,:);

N_z = [length(z{1}), length(z{2})];
for t = 1:T
    % Decompose mixture spectrogram
    mixture_act = fit_plca_mixture(V(:,t), z{1}, dict{1}, iter_max);
    % Compute the approximation based on the previous decomposition
    approx = sum(dict{1}.freq_basis .* repmat(mixture_act,[F,1]),2);
    d_kl_mixture = KLDiv(P_V(:,t)', approx');
    
    if d_kl_mixture < theta_kl
        % Supervised separation
        input_dict = dict;
        input_dict{2}.temp_act = dict{2}.temp_act(t,:);
        out_temp = fit_plca_supervised(V(:,t), z, input_dict, iter_max);
        temp_act(t,:) = out_temp(1:N_z(1));
        dict{2}.temp_act(t,:) = out_temp(N_z(1)+1:sum(N_z));
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
    clc
    disp(['t: ',num2str(t),' / ',num2str(T)])
end

%% Reconstruct the sources and store the results
for s = 1:2
    spectrogram_dict = rec_spectrogram(:,:,s) .* phase_spect;

    S = spectrogram_dict;
    out(1:Nshift*(T-1)+N_window,1)=0;

    for i=1:T
        Sfft=[S(:,i);conj(S(N_window/2:-1:2,i))];
        stemp=real(ifft(Sfft));
        sout=stemp(1:N_window,1) .* sqrt(Window);

        out(Nshift*(i-1)+1:Nshift*(i-1)+N_window,1)=out(Nshift*(i-1)+1:Nshift*(i-1)+N_window,1)+sout;
    end

    audiowrite(['Prototypes/Separation/Results/source_',num2str(s),'_reconstruct.wav'],out,fs)
end