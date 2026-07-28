clear
clc
% addpath(genpath('Toolbox'))
% addpath(genpath('Prototypes/Separation'))
addpath(genpath(pwd))
%distcomp.feature( 'LocalUseMpiexec', false )
%% Load data
%band = '051 - AM Contra - Heart Peripheral';
band = '052 - ANiMAL - Easy Tiger';
%band = '053 - Actions - Devil''s Words';
%band = '094 - Titanium - Haunted Age';
[mixture,fs] = audioread(['..\Data\MUS2016\DSD100\Mixtures\Dev\',band,'\mixture.wav']);
source_1 = audioread(['..\Data\MUS2016\DSD100\Sources\Dev\',band,'\bass.wav']);
source_2 = audioread(['..\Data\MUS2016\DSD100\Sources\Dev\',band,'\drums.wav']);
source_3 = audioread(['..\Data\MUS2016\DSD100\Sources\Dev\',band,'\other.wav']);
source_4 = audioread(['..\Data\MUS2016\DSD100\Sources\Dev\',band,'\vocals.wav']);
% 
% [mixture,fs] = audioread(['..\Data\Shannon\mixture.wav']);
% source_1 = audioread(['..\Data\Shannon\drums.wav']);
% source_2 = audioread(['..\Data\Shannon\voice.wav']);
% source_3 = audioread(['..\Data\Shannon\others.wav']);

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

% start_time = 15;
% end_time = 25;
start_time = 20;
end_time = 30;
% start_time = 35;
% end_time = 45;
source_1 = source_1(:,start_time*fs:end_time*fs);
source_2 = source_2(:,start_time*fs:end_time*fs);
source_3 = source_3(:,start_time*fs:end_time*fs);
source_4 = source_4(:,start_time*fs:end_time*fs);
source{1} = source_1;
source{2} = source_2;
source{3} = source_3;
source{4} = source_4;

mixture = mixture(:,start_time*fs:end_time*fs);
%%
path='\\AUDIO06\ServeurMatlab\Ph. D Thesis\Data\Music\cut\';
song = 'stereophonic';
[mixture,fs] = audioread([path,song,'.wav']);
if size(mixture,1) > size(mixture,2)
    mixture = mixture';
end
%
P_f = [];
Z = [];
n_clusters = [20,60,20,20];
file = load(['Prototypes/Separation/Data/dict_drums_acoustic.mat']);
%[~,C] = kmeans(file.V,50);
[~,C] = kmedoids(file.V,n_clusters(1));
P_f = [P_f; C];
Z(1) = size(C,1);
file = load(['Prototypes/Separation/Data/dict_vocals_acoustic.mat']);
%[~,C] = kmedoids(file.V,50);
[~,C] = kmedoids(file.V,n_clusters(2));
P_f = [P_f; C];
Z(2) = size(C,1);
file = load(['Prototypes/Separation/Data/dict_guitar_acoustic.mat']);
%[~,C] = kmeans(file.V,50);
[~,C] = kmedoids(file.V,n_clusters(3));
P_f = [P_f; C];
Z(3) = size(C,1);
file = load(['Prototypes/Separation/Data/dict_bass_acoustic.mat']);
%[~,C] = kmeans(file.V,50);
[~,C] = kmedoids(file.V,n_clusters(4));
P_f = [P_f; C];
Z(4) = size(C,1);
% Compute spectrogram
N_window = 1024;
Window = hanning(N_window);
Nshift = N_window / 2;
Nfft = N_window;
spectrogram_complex = [];
for ch = 1:2
    spectrogram_complex(:,:,ch) = spectrogram(mixture(ch,:),sqrt(Window),Nshift,Nfft);
end

V = abs(spectrogram_complex);
phase_spect = angle(spectrogram_complex);

[F,T,I] = size(V);
% figure(1)
% imagesc(db(V))
% Online new
iter_max = 50;
n_source = length(Z);
% Fit dictionary
R = 1;
rec_spectrogram = zeros(F,T,I,n_source);
cum_Z = [0,cumsum(Z)];
mask_old = ones(F,R,I,n_source);
alpha = 1 - exp(-3*1024/(48000*0.064));
tic
for t = R:T
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
    mask = alpha * num ./ repmat(denom,[1,1,1,n_source]) + (1-alpha)*mask_old;

    for s = 1:n_source
        rec_spectrogram(:,t,:,s) = buffers(:,end,:) .* mask(:,end,:,s);
    end
    mask_old = mask;
    progressbar(t,T)
end
elapsed_time = toc;
% Reconstruct the sources and store the results
%classes = {'Bass','Drums','Guitar','Vocals'};
classes = {'Drums','Vocals','Guitar','Bass'};
out = [];
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

    audiowrite(['Prototypes/Separation/Results/PLCS/',song,'_',classes{s},'_reconstruct.wav'],out,fs)
end


%% Unsupervised PLCS
Z = 4;
iter_max = 200;
fixed = {[],[],[]};
% Fit dictionary
[P_z,P_f,P_t] = plcs(V,Z,iter_max,fixed);

num = zeros(F,T,I,Z);
for s = 1:Z
    for i = 1:I
        aux = repmat(reshape(P_z(s,i) .* P_t(s,:),[1,1,T]),[1,F,1]);
        P_zft = aux .* repmat(P_f(s,:),[1,1,T]);
        num(:,:,i,s) = reshape(sum(P_zft,1),[F,T]);
    end
end   

denom = sum(num,4);
mask = num ./ repmat(denom,[1,1,1,Z]);

rec_spectrogram = zeros(F,T,I,Z);
for s = 1:Z
    rec_spectrogram(:,:,:,s) = V .* mask(:,:,:,s);
end

for s = 1:Z
    spectrogram_dict = rec_spectrogram(:,:,:,s) .* exp(1i * phase_spect);
    
    out(1:Nshift*(T-1)+N_window,1) = 0;
    out(1:Nshift*(T-1)+N_window,2) = 0;
    for i = 1:I
        S = spectrogram_dict(:,:,i);

        for t = 1:T
            Sfft = [S(:,t);conj(S(N_window/2:-1:2,t))];
            stemp = real(ifft(Sfft));
            sout = stemp(1:N_window,1) .* sqrt(Window);

            out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,i) = out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,i)+sout;
        end
    end
    audiowrite(['Prototypes/Separation/Results/source_',num2str(s),'_reconstruct.wav'],out,fs)
end
%% Unsupervised PLCA on each source
n_source = 4;
Z_1 = 5;
iter_max = 100;
fixed = {[],[],[]};
P_z = [];
P_f = [];
P_t = [];
Z = zeros(1,n_source);
for s = 1:n_source
%     if s == 4
%         load('Prototypes/Separation/Data/dict_vocals_plca.mat')
%         P_f = [P_f;V];
%         Z(s) = Z_1;
%     else
%         for ch = 1:2
%             spectrogram_source(:,:,ch) = spectrogram(source{s}(ch,:),sqrt(Window),Nshift,Nfft);
%         end
%         V_source = abs(spectrogram_source);
%         % Fit dictionary
%         idx_min = Z_1;
%         [aux_z,aux_f,aux_t] = plcs(V_source,idx_min,iter_max,fixed);
%         P_z = [P_z; aux_z];
%         P_f = [P_f; aux_f];
%         P_t = [P_t; aux_t];
%         Z(s) = idx_min;
%    end
    if s == 1
        load('Prototypes/Separation/Data/dict_bass_plca_r.mat')
    elseif s == 2
        load('Prototypes/Separation/Data/dict_drums_plca_r.mat')
    elseif s == 3
        load('Prototypes/Separation/Data/dict_other_plca_r.mat')
    elseif s == 4
        file = load('Prototypes/Separation/Data/dict_vocals_plcs.mat');
        rV = file.P_f';
    end
    %P_f = [P_f;V];
    %Z(s) = size(V,1);
    P_f = [P_f; rV'];
    Z(s) = size(rV,2);
    progressbar(s,n_source)
end

%% Online
iter_max = 10;
% Fit dictionary
R = 1;
rec_spectrogram = zeros(F,T,I,n_source);
cum_Z = [0,cumsum(Z)];
tic
for t = R:T
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
% Reconstruct the sources and store the results
classes = {'Bass','Drums','Others','Vocals'};
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

    audiowrite(['Prototypes/Separation/Results/',classes{s},'_reconstruct.wav'],out,fs)
end

%% 
n_source = 2;
%Z_1 = 20;
P_f = [];
fixed = {[],[],[]};
iter_max = 200;
%Z = zeros(1,n_source);
Z = [50,50];
for s = 1:n_source
    if s == 1
        file = load('Prototypes/Separation/Data/dict_vocals_plcs.mat');
%         spectrogram_source = [];
%         for ch = 1:2
%             spectrogram_source(:,:,ch) = spectrogram(source{4}(ch,:),sqrt(Window),Nshift,Nfft);
%         end
%         V_source = abs(spectrogram_source);
%         % Fit dictionary
%         [~,rV,~] = plcs(V_source,Z(1),iter_max,fixed);
%         rV = rV';
         rV = file.P_f';
         
    else
        rV = rand(F,Z(2));
        rV = rV ./ repmat(sum(rV),[F,1]);
    end
    %P_f = [P_f;V];
    %Z(s) = size(V,1);
    P_f = [P_f; rV'];
    Z(s) = size(rV,2);
    progressbar(s,n_source)
end
%% Semi-supervised
iter_max = 10;
% Fit dictionary
R = 10;
rec_spectrogram = zeros(F,T,I,n_source);
cum_Z = [0,cumsum(Z)];
id_upd = cum_Z(2)+1:cum_Z(end);
tic
for t = 1:R
    for s = 1:n_source
        rec_spectrogram(:,t,:,s) = V(:,t,:);
    end
end
P_fn = P_f;

P_s = rand(n_source,R,I);
P_t = rand(cum_Z(end),R);
for t = R:T
    buffers = V(:,t-R+1:t,:);
    
    %[P_s,P_fn,P_t] = plcs_semiseparation(buffers,Z,iter_max,P_fn,id_upd);

    rV = repmat(reshape(buffers,[1,F,R,I]),[cum_Z(end),1,1,1]);
    for iter = 1:iter_max
        % E-Step
        P_szft = zeros(cum_Z(end),F,R,I);
        norm_coeff = zeros(F,R,I);
        for s = 1:n_source
            aux = repmat(reshape(P_t(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),1,R]),[1,F,1]) .* repmat(reshape(P_fn(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),F,1]),[1,1,R]);
            for i = 1:I
                P_szft(cum_Z(s)+1:cum_Z(s+1),:,:,i) = repmat(reshape(P_s(s,:,i),[1,1,R]),[Z(s),F,1]) .* aux;
                norm_coeff(:,:,i) = norm_coeff(:,:,i) + reshape(repmat(reshape(P_s(s,:,i),[1,1,R]),[1,F,1]) .* sum(aux,1),[F,R]);
            end
        end
        P_szft = P_szft ./ repmat(reshape(norm_coeff,[1,F,R,I]),[cum_Z(end),1,1,1]);

        % M-step
        VP = rV .* P_szft;
        VP2 = reshape(sum(VP,2),[cum_Z(end),R,I]);
        VP2ch = sum(VP2,3);
        VPch = sum(VP,4);
        VP3 = sum(VPch,3);
        VP32 = zeros(n_source,R,I);
        for s = 1:n_source
            VP32(s,:,:) = sum(VP2(cum_Z(s)+1:cum_Z(s+1),:,:),1);
        end
        VP32ch = sum(VP32,3);

        P_s = VP32;
        P_s = P_s ./ repmat(sum(P_s,1),[n_source,1,1]);

        P_fn(id_upd,:) = VP3(id_upd,:) ./ repmat(sum(VP3(id_upd,:),2),[1,F]);


        for s = 1:n_source
            P_t(cum_Z(s)+1:cum_Z(s+1),:) = VP2ch(cum_Z(s)+1:cum_Z(s+1),:) ./ repmat(VP32ch(s,:),[Z(s),1]);
        end
        %progressbar(iter,iter_max)
    end
    
%     figure(1)
%     clf
%     subplot(2,1,1)
%     plot(P_fn(id_upd,:)')
%     subplot(2,1,2)
%     plot(P_fn(1:20,:)')
%     pause(0.0001)

    num = zeros(F,R,I,n_source);
    for s = 1:n_source
        aux = repmat(reshape(P_t(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),1,R]),[1,F,1]) .* repmat(reshape(P_fn(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),F,1]),[1,1,R]);
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
% Reconstruct the sources and store the results
classes = {'Vocals','Others'};
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

    audiowrite(['Prototypes/Separation/Results/',classes{s},'_reconstruct.wav'],out,fs)
end

