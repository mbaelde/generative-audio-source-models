clear
clc
addpath(genpath(pwd))

%% Load data and learn PLCA for every sound
data_folder = 'F:\Projets\Drums hits detection\mapex\';
class = dir(data_folder);
class = class(3:end);
n_class = length(class);

% Spectrogram parameters
N_window = 1024;
Window = hanning(N_window);
Nshift = N_window / 2;
Nfft = N_window;

% PLCA parameters
Z = 10*ones(1,n_class);%[20,10,1,5,5,1,2,5,1,10,20,20,20,20,20,1,1,1,1,1];
iter_max = 200;
fixed = {[],[],[]};

P_f = [];
for k = 1:n_class
    clc
    disp(['class: ',class(k).name, ' (',num2str(k),' / ',num2str(n_class),')\n'])
    [source,fs] = audioread([data_folder,class(k).name,'/',class(k).name,'_mf.wav']);
    % Transpose to get n_channel x n_samples
    if size(source,1) > size(source,2)
        source = source';
    end
    if size(source,1) > 1
        source = mean(source);
    end
    
    spectrogram_complex = spectrogram(source,sqrt(Window),Nshift,Nfft);
    V = abs(spectrogram_complex);
    [F,T] = size(V);
    
    [~,aux_P_f,~] = plca(V,Z(k),iter_max,fixed);
    P_f = [P_f; aux_P_f];
end

save([data_folder,'..\P_f.mat'],'P_f')

%% Load mixture sound
load([data_folder,'..\P_f.mat'],'P_f')
music = 'superMassive';
[mixture,fs] = audioread([data_folder,'..\drumstikAudioSamples\drumstik_',music,'_100.wav']);
% Transpose to get n_channel x n_samples
if size(mixture,1) > size(mixture,2)
    mixture = mixture';
end
% Mean to get mono channel
if size(mixture,1) > 1
    mixture = mean(mixture);
end

spectrogram_complex = spectrogram(mixture,sqrt(Window),Nshift,Nfft);
V = abs(spectrogram_complex);
[F,T] = size(V);
phase_spect = angle(spectrogram_complex);

%% Online supervised PLCA
iter_max = 50;
R = 1;
rec_spectrogram = zeros(F,T,n_class);

cum_Z = [0,cumsum(Z)];
tic
for t = R:T
    clc
    disp(['t: ',num2str(t),' / ',num2str(T)])
    buffers = V(:,t-R+1:t);
    [P_s,P_t] = plca_separation(buffers,Z,iter_max,P_f);

    num = zeros(F,R,n_class);
    for s = 1:n_class
        aux = repmat(reshape(P_t(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),1,R]),[1,F,1]) .* repmat(reshape(P_f(cum_Z(s)+1:cum_Z(s+1),:),[Z(s),F,1]),[1,1,R]);
        num(:,:,s) = repmat(P_s(s,:),[F,1]) .* reshape(sum(aux,1),[F,R]);
    end
    denom = sum(num,3);
    mask = num ./ repmat(denom,[1,1,n_class]);

    for s = 1:n_class
        rec_spectrogram(:,t,s) = buffers(:,end) .* mask(:,end,s);
    end
   %progressbar(t,T)
end
elapsed_time = toc;
% Reconstruct the sources and store the results
for s = 1:n_class
    spectrogram_dict = rec_spectrogram(:,:,s) .* exp(1i * phase_spect);
    
    S = spectrogram_dict;
    out(1:Nshift*(T-1)+N_window,1)=0;

    for t = 1:T
        Sfft = [S(:,t);conj(S(N_window/2:-1:2,t))];
        stemp = real(ifft(Sfft));
        sout = stemp(1:N_window,1) .* sqrt(Window);

        out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1) = out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)+sout;
    end
    audiowrite(['F:\Projets\Drums hits detection\Results_PLCA\',music,'\',class(s).name,'_reconstruct.wav'],out,fs)
end


%% Unsupervised PLCA
Z_1 = 3;
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
    audiowrite(['Prototypes/Separation/Results/source_',num2str(s),'_reconstruct.wav'],out,fs)
end