clear
clc

addpath(genpath(pwd))

%% Load data
file_folder = '\\AUDIO06\ServeurMatlab\Ph. D Thesis\Data\BF\';
file_name = 'audio_bf_no.wav';

[mixture,fs] = audioread([file_folder,file_name]);

% Transpose to get n_channel x n_samples
if size(mixture,1) > size(mixture,2)
    mixture = mixture';
end
% Mean to get mono channel
if size(mixture,1) > 1
    mixture = mean(mixture);
end
I = size(mixture,1);
%% Compute spectrogram
N_window = 2048;
Window = hanning(N_window);
Nshift = N_window / 2;
Nfft = N_window;
for i = 1:I
    spectrogram_complex(:,:,i) = spectrogram(mixture(i,:),sqrt(Window),Nshift,Nfft);
end
V = abs(spectrogram_complex);
phase_spect = angle(spectrogram_complex);

[F,T,I] = size(V);
% figure(1)
% imagesc(db(V))
%% Unsupervised PLCA
Z_1 = 4;
iter_max = 100;
fixed = {[],[],[]};
% Fit dictionary
[P_z,P_f,P_t] = plca(V,Z_1,iter_max,fixed);

num = zeros(F,T,Z_1);
for s = 1:Z_1
    aux = repmat(reshape(P_z(s) .* P_t(s,:),[1,1,T]),[1,F,1]);
    P_zft = aux .* repmat(P_f(s,:),[1,1,T]);
    num(:,:,s) = reshape(sum(P_zft,1),[F,T]);
end
denom = sum(num,3);
mask = num ./ repmat(denom,[1,1,Z_1]);
rec_spectrogram = zeros(F,T,Z_1);
for s = 1:Z_1
    rec_spectrogram(:,:,s) = V .* mask(:,:,s);
end

for s = 1:Z_1
    spectrogram_dict = rec_spectrogram(:,:,s) .* exp(1i * phase_spect);
    
    S = spectrogram_dict;
    out(1:Nshift*(T-1)+N_window,1)=0;
    
    for t = 1:T
        Sfft = [S(:,t);conj(S(N_window/2:-1:2,t))];
        stemp = real(ifft(Sfft));
        sout = stemp(1:N_window,1) .* sqrt(Window);
        
        out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1) = out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)+sout;
    end
    %     if s == 2
    %         out = out * 10^(12/20);
    %     end
    audiowrite(['Prototypes/Separation/Results/BF/source_',num2str(s),'_reconstruct.wav'],out,fs)
end
%% Load preprocess dictionaries
load('Database/A-Volute/Complete/feature_fs44100_N1025.mat')
fold = 1;
n_class = 5;
load('Database/A-Volute/Folds/Set 1/idx_train_test_fs44100_T2048.mat', 'idx_train', 'idx_test')
load('Database/A-Volute/Folds/Set 1/Z_fold_1_fs44100_T2048.mat', 'Z')
feature_training = [];
for k = 1:n_class
    feature_class = feature(feature(:,end-1) == k,:);
    feature_training = [feature_training; feature_class(idx_train{fold}{k},:)];
end
clear feature_class

N_sounds = [29   152   163    53    87]; % 96%
P_f = reduce_dictionary(feature_training, Z, N_sounds);
clear Z
Z = zeros(1,n_class);
for k = 1:n_class
    Z(k) = sum(P_f(:,end-1) == k);
end
P_f = P_f(:,1:end-2);
%% Online PLCA
n_source = n_class;
iter_max = 25;
% Fit dictionary
R = 1;
rec_spectrogram = zeros(F,T,n_source);
cum_Z = [0,cumsum(Z)];
tic
for t = 1:T
    buffers = V(:,t-R+1:t);
    [P_s,P_t] = plca_separation(buffers,Z,iter_max,P_f);
 
    num = zeros(F,R,n_source);
    for s = 1:n_source
        aux = repmat(reshape(P_t(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),1,R]),[1,F,1]) .* repmat(reshape(P_f(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),F,1]),[1,1,R]);
        num(:,:,s) = repmat(P_s(s,:),[F,1]) .* reshape(sum(aux,1),[F,R]);
    end
    denom = sum(num,3);
    mask = num ./ repmat(denom,[1,1,n_source]);

    for s = 1:n_source
        rec_spectrogram(:,t,s) = buffers(:,end) .* mask(:,end,s);
    end
    progressbar(t,T)
end
elapsed_time = toc;
%% Reconstruct the sources and store the results (plca)
classes = {'alarm','detonation','engine','step','voice'};
out = [];
for s = 1:n_source
    spectrogram_dict = rec_spectrogram(:,:,s) .* exp(1i * phase_spect);
    
    out(1:Nshift*(T-1)+N_window,1)=0;

    S = spectrogram_dict;

    for t = 1:T
        Sfft = [S(:,t);conj(S(N_window/2:-1:2,t))];
        stemp = real(ifft(Sfft));
        sout = stemp(1:N_window,1) .* sqrt(Window);

        out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1) = out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)+sout;
    end

    audiowrite(['Prototypes/Separation/Results/BF/',classes{s},'_reconstruct.wav'],out,fs)
end
%% Online PLCS
n_source = n_class;
iter_max = 50;
% Fit dictionary
R = 1;
rec_spectrogram = zeros(F,T,I,n_source);
cum_Z = [0,cumsum(Z)];
tic
for t = 1:T
    buffers = V(:,t-R+1:t,:);
    [P_s,P_t] = plcs_separation(buffers,Z,iter_max,P_f);
 
    num = zeros(F,R,I,n_source);
    for s = 1:n_source
        aux = repmat(reshape(P_t(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),1,R]),[1,F,1]) .* repmat(reshape(P_f(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),F,1]),[1,1,R]);
        for i = 1:I
            num(:,:,i,s) = repmat(P_s(s,:,i),[F,1]) .* reshape(sum(aux,1),[F,R]);
        end
    end
    denom = sum(num,4);
    mask = num ./ repmat(denom,[1,1,1,n_source]);

    for s = 1:n_source
        rec_spectrogram(:,t,:,s) = buffers(:,end,:) .* mask(:,end,:,s);
    end
    progressbar(t,T)
end
elapsed_time = toc;
%% Reconstruct the sources and store the results (plcs)
classes = {'alarm','detonation','engine','step','voice'};
for s = 1:n_source
     spectrogram_dict = rec_spectrogram(:,:,:,s) .* exp(1i * phase_spect);
    
    out(1:Nshift*(T-1)+N_window,1)=0;
    out(1:Nshift*(T-1)+N_window,2)=0;
    for i = 1:I
        S = spectrogram_dict(:,:,i);

        for t = 1:T
            Sfft = [S(:,t);conj(S(N_window/2:-1:2,t))];
            stemp = real(ifft(Sfft));
            sout = stemp(1:N_window,1) .* sqrt(Window);

            out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,i) = out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,i)+sout;
        end
    end
    audiowrite(['Prototypes/Separation/Results/BF/',classes{s},'_reconstruct.wav'],out,fs)
end
%% Learn PLCA for the voice
x = source(4,:);
x = mixture;
spectro_voice = spectrogram(x,sqrt(Window),Nshift,Nfft);
V_voice = abs(spectro_voice);

Z_1 = 60;
Z_1 = 5;
iter_max = 500;
fixed = {[],[],[]};
% Fit dictionary
[P_z,P_f,P_t] = plca(V,Z_1,iter_max,fixed);
% P_fvoice = P_f;
% save('Prototypes/Separation/Results/dict_voice_plca.mat','P_f')
% P_fvoice = load('Prototypes/Separation/Results/dict_voice_plca.mat','P_f');
% P_fvoice = P_fvoice.P_f;

%% Semi-supervised separation
% Mixture spectrogram
energy_v = sum(V,1);
P_V = V ./ repmat(energy_v,[F,1]);
% Source 2 dictionary
Z_2 = 100;
% EM parameters
iter_max = 100;
% Reconstructed spectrograms
rec_spectrogram = zeros(F,T,2);
N_Z = Z_1 + Z_2;

% Perform PLCA
P_z = rand(N_Z,1);
P_z = P_z ./ sum(P_z);
% P_z(1:Z_1) = P_z(1:Z_1) ./ sum(P_z(1:Z_1));
% P_z(Z_1+1:N_Z) = P_z(Z_1+1:N_Z) ./ sum(P_z(Z_1+1:N_Z));

P_f = rand(Z_2,F);
P_f = P_f ./ repmat(sum(P_f,2),[1,F]);
P_f = [P_fvoice; P_f];

P_t = rand(N_Z,T);
P_t = P_t ./ repmat(sum(P_t,2),[1,T]);

% rV_1 = repmat(reshape(V,[1,F,T]),[Z_1,1,1]);
% rV_2 = repmat(reshape(V,[1,F,T]),[Z_2,1,1]);
rV = repmat(reshape(V,[1,F,T]),[N_Z,1,1]);
for iter = 1:iter_max
    % E-Step
    %     aux = repmat(reshape((P_z(1:Z_1) * ones(1,T)) .* P_t(1:Z_1,:),[Z_1,1,T]),[1,F,1]);
    %     P_zft_1 = aux .* repmat(P_f(1:Z_1,:),[1,1,T]);
    %     P_zft_1 = P_zft_1 ./ repmat(sum(P_zft_1,1),[Z_1,1,1]);
    %
    %     aux = repmat(reshape((P_z(Z_1+1:N_Z) * ones(1,T)) .* P_t(Z_1+1:N_Z,:),[Z_2,1,T]),[1,F,1]);
    %     P_zft_2 = aux .* repmat(P_f(Z_1+1:N_Z,:),[1,1,T]);
    %     P_zft_2 = P_zft_2 ./ repmat(sum(P_zft_2,1),[Z_2,1,1]);
    aux = repmat(reshape((P_z * ones(1,T)) .* P_t,[N_Z,1,T]),[1,F,1]);
    P_zft = aux .* repmat(P_f,[1,1,T]);
    P_zft = P_zft ./ repmat(sum(P_zft,1),[N_Z,1,1]);
    
    % M-step
    %     VP_1 = rV_1 .* P_zft_1;
    %     VP_2 = rV_2 .* P_zft_2;
    VP = rV .* P_zft;
    %     VP2_1 = reshape(sum(VP(1:Z_1,:,:),2),[Z_1,T]);
    %     VP2_2 = reshape(sum(VP(Z_1:N_Z,:,:),2),[Z_2,T]);
    VP2 = reshape(sum(VP,2),[N_Z,T]);
    %     VP3_1 = sum(VP(1:Z_1,:,:),3);
    %     VP3_2 = sum(VP(Z_1:N_Z,:,:),3);
    VP3 = sum(VP,3);
    %     VP32_1 = sum(VP3(1:Z_1,:,:),2);
    %     VP32_2 = sum(VP3(Z_1:N_Z,:,:),2);
    VP32 = sum(VP3,2);
    
    %     P_z1 = VP32_1;
    %     P_z1 = P_z1 ./ sum(P_z1);
    %     P_z2 = VP32_2;
    %     P_z2 = P_z2 ./ sum(P_z2);
    %     P_z = [P_z1; P_z2];
    P_z = VP32;
    P_z = P_z ./ sum(P_z);
    
    P_f2 = VP3(Z_1+1:N_Z,:) ./ repmat(VP32(Z_1+1:N_Z,:),[1,F]);
    P_f = [P_fvoice; P_f2];
    
    %     P_t1 = VP2_1 ./ repmat(VP32_1,[1,T]);
    %     P_t2 = VP2_2 ./ repmat(VP32_2,[1,T]);
    %     P_t = [P_t1; P_t2];
    P_t = VP2 ./ repmat(VP32,[1,T]);
    
    progressbar(iter,iter_max)
end

% Reconstruct sources
aux = repmat(reshape((P_z(1:Z_1) * ones(1,T)) .* P_t(1:Z_1,:),[Z_1,1,T]),[1,F,1]);
P_zft_1 = aux .* repmat(P_f(1:Z_1,:),[1,1,T]);
aux = repmat(reshape((P_z(Z_1+1:N_Z) * ones(1,T)) .* P_t(Z_1+1:N_Z,:),[Z_2,1,T]),[1,F,1]);
P_zft_2 = aux .* repmat(P_f(Z_1+1:N_Z,:),[1,1,T]);
num(:,:,1) = reshape(sum(P_zft_1,1),[F,T]);
num(:,:,2) = reshape(sum(P_zft_2,1),[F,T]);
denom = sum(num,3);
rec_spectrogram(:,:,1) = V .* num(:,:,1) ./ denom;
rec_spectrogram(:,:,2) = V .* num(:,:,2) ./ denom;