clear
clc

addpath(genpath(pwd))

%% Load data
file_folder = '\\AUDIO06\ServeurMatlab\Ph. D Thesis\Data\';
type = 'Music\';
file_name = 'bad_romance.wav';

[mixture,fs] = audioread([file_folder,type,file_name]);

% Transpose to get n_channel x n_samples
if size(mixture,1) > size(mixture,2)
    mixture = mixture';
end
% Mean to get mono channel
if size(mixture,1) > 1
    mixture = mean(mixture);
end

%mixture = mixture(1:round(length(mixture)/10));
% Compute spectrogram
N_window = 2048;
Window = hanning(N_window);
Nshift = N_window / 2;
Nfft = N_window;
spectrogram_complex = spectrogram(mixture,sqrt(Window),Nshift,Nfft);

V = abs(spectrogram_complex);
phase_spect = angle(spectrogram_complex);

[F,T] = size(V);
% figure(1)
% imagesc(db(V))
% Unsupervised PLCA
% Z_1 = 5;
% iter_max = 100;
% fixed = {[],[],[]};
% % Fit dictionary
% [P_z,P_f,P_t] = plca(V,Z_1,iter_max,fixed);
% 
% num = zeros(F,T,Z_1);
% for s = 1:Z_1
%     aux = repmat(reshape(P_z(s) .* P_t(s,:),[1,1,T]),[1,F,1]);
%     P_zft = aux .* repmat(P_f(s,:),[1,1,T]);
%     num(:,:,s) = reshape(sum(P_zft,1),[F,T]);
% end
% denom = sum(num,3);
% mask = num ./ repmat(denom,[1,1,Z_1]);
% rec_spectrogram = zeros(F,T,Z_1);
% for s = 1:Z_1
%     rec_spectrogram(:,:,s) = V .* mask(:,:,s);
% end
% 
% for s = 1:Z_1
%     spectrogram_dict = rec_spectrogram(:,:,s) .* exp(1i * phase_spect);
%     
%     S = spectrogram_dict;
%     out(1:Nshift*(T-1)+N_window,1)=0;
%     
%     for t = 1:T
%         Sfft = [S(:,t);conj(S(N_window/2:-1:2,t))];
%         stemp = real(ifft(Sfft));
%         sout = stemp(1:N_window,1) .* sqrt(Window);
%         
%         out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1) = out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)+sout;
%     end
%     %     if s == 2
%     %         out = out * 10^(12/20);
%     %     end
%     audiowrite(['Prototypes/Separation/Results/Movie/source_',num2str(s),'_reconstruct.wav'],out,fs)
% end
% Load dictionary
load('Database/A-Volute/Complete/feature_fs44100_N1025.mat')
fold = 1;
n_class = 5;
load('Database/A-Volute/Folds/Set 1/idx_train_test_fs44100_T2048.mat', 'idx_train', 'idx_test')

feature_training = [];
for k = 1:n_class
    feature_class = feature(feature(:,end-1) == k,:);
%     if k < 5
%         N_sounds(k) = size(feature_class,1) / 10;
%     elseif k == 5
%         N_sounds(k) = size(feature_class,1) / 150;
%     end
    feature_training = [feature_training; feature_class(idx_train{fold}{k},:)];
end
clear feature_class

N_sounds = [66   189   221    80   145];% 95%;
load('Database/A-Volute/Folds/Set 1/Z_fold_1_fs44100_T2048.mat', 'Z')
P_f = reduce_dictionary(feature_training, Z, N_sounds);

alarm = P_f(P_f(:,end-1) == 1,:);
voice = P_f(P_f(:,end-1) == 5,:);
voice(:,end-1) = 1;
other = [P_f(P_f(:,end-1) == 2,:);P_f(P_f(:,end-1) == 3,:);P_f(P_f(:,end-1) == 4,:);];

P_f = [alarm;voice;other];

clear Z
Z = zeros(1,4);
for k = 1:4
    Z(k) = sum(P_f(:,end-1) == k);
end
P_f = P_f(:,1:end-2);
cum_Z = [0,cumsum(Z)];

% Online
iter_max = 50;
n_source = 4;
% Fit dictionary
R = 1;
rec_spectrogram = zeros(F,T,n_source);

tic
for t = R:T
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
% Reconstruct the sources and store the results
classes = {'voice','detonation','engine','step'};
for s = 1:n_source
    spectrogram_dict = rec_spectrogram(:,:,s) .* exp(1i * phase_spect);
    
    S = spectrogram_dict;
    out(1:Nshift*(T-1)+N_window,1)=0;
    
    for t = 1:T
        Sfft = [S(:,t);conj(S(N_window/2:-1:2,t))];
        stemp = real(ifft(Sfft));
        sout = stemp(1:N_window,1) .* sqrt(Window);
        
        out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1) = out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)+sout;
    end

    audiowrite(['Prototypes/Separation/Results/',type,file_name(1:end-4),'_',classes{s},'_reconstruct.wav'],out,fs)
end
