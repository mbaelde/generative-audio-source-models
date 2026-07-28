clear
clc
addpath('bss_eval')
addpath('Statistics')
%% Load data
type = 'video_game';

if strcmp(type,'general')
    [source_1,fs_1] = audioread('laugh_1.wav');
    [source_2,fs_2] = audioread('dog_1.wav');
    [mixture,fs] = audioread('mixture_dog_laugh.wav');
elseif strcmp(type,'video_game')
    [source_1,fs_1] = audioread('gunshot_3.wav');
    [source_2,fs_2] = audioread('airplane_1.wav');
    %[source_2,fs_2] = audioread('voice_1.wav');
    [mixture,fs] = audioread('mixture_airplane_gunshot.wav');
    %[mixture,fs] = audioread('mixture_gunshot_voice_2.wav');
elseif strcmp(type,'denoising')
    [mixture,fs] = audioread('mixture_voice_noise.wav');
    source_1 = zeros(length(mixture),1);
    source_2 = zeros(length(mixture),1);
elseif strcmp(type,'music')
    [mixture,fs] = audioread('F:\Ph. D Thesis\Experimental Softwares\Lex2\Data (Mixture)\MUS2016\DSD100\Mixtures\Dev\052 - ANiMAL - Easy Tiger\mixture.wav');
    source_1 = audioread('F:\Ph. D Thesis\Experimental Softwares\Lex2\Data (Mixture)\MUS2016\DSD100\Sources\Dev\052 - ANiMAL - Easy Tiger\bass.wav');
    source_2 = audioread('F:\Ph. D Thesis\Experimental Softwares\Lex2\Data (Mixture)\MUS2016\DSD100\Sources\Dev\052 - ANiMAL - Easy Tiger\drums.wav');
    source_3 = audioread('F:\Ph. D Thesis\Experimental Softwares\Lex2\Data (Mixture)\MUS2016\DSD100\Sources\Dev\052 - ANiMAL - Easy Tiger\other.wav');
    source_4 = audioread('F:\Ph. D Thesis\Experimental Softwares\Lex2\Data (Mixture)\MUS2016\DSD100\Sources\Dev\052 - ANiMAL - Easy Tiger\vocals.wav');
end

if size(source_1,1) > size(source_1,2)
    source_1 = source_1';
end
if size(source_2,1) > size(source_2,2)
    source_2 = source_2';
end
if strcmp(type,'music')
    if size(source_3,1) > size(source_3,2)
        source_3 = source_3';
    end
    if size(source_4,1) > size(source_4,2)
        source_4 = source_4';
    end
end
if size(mixture,1) > size(mixture,2)
    mixture = mixture';
end

if size(source_1,1) > 1
    source_1 = mean(source_1);
end
if size(source_2,1) > 1
    source_2 = mean(source_2);
end
if strcmp(type,'music')
    if size(source_3,1) > 1
        source_3 = mean(source_3);
    end
    if size(source_4,1) > 1
        source_4 = mean(source_4);
    end
end
if size(mixture,1) > 1
    mixture = mean(mixture);
end

N_1 = length(source_1);
N_2 = length(source_2);

if N_1 < N_2
    source_2 = source_2(1:N_1);
    mixture = mixture(1:N_1);
elseif N_1 > N_2
    source_1 = source_1(1:N_2);
    mixture = mixture(1:N_2);
end

source = [source_1;source_2];
if strcmp(type,'music')
    source = [source_1;source_2;source_3;source_4];
end
%% Compute spectrogram
N_window = 2048;
Window = hanning(N_window);
Nshift = N_window / 2;
Nfft = N_window;
spectrogram_complex = spectrogram(mixture,sqrt(Window),Nshift,Nfft);

factor = 1;
V = factor*abs(spectrogram_complex);
phase_spect = angle(spectrogram_complex);

[F,T] = size(V);
figure(1)
imagesc(db(V))

%% Perform PLCA
Z = 2;
iter_max = 300;
fixed = {[],[],[]};
[P_z,P_f,P_t] = plca(V,Z,iter_max,fixed);

% Recover sources
mask = zeros(F,T,Z);
for z = 1:Z
    mask(:,:,z) = P_z(z) * P_f(z,:)' * P_t(z,:);
end
mask = mask ./ repmat(sum(mask,3),[1,1,Z]);

spectrogram_source = zeros(F,T,Z);
for z = 1:Z
    spectrogram_source(:,:,z) = ((V./factor) .* mask(:,:,z)) .* exp(1i * phase_spect);
end

figure(2)
imagesc(db(abs(spectrogram_source(:,:,1))))

figure(3)
imagesc(db(abs(spectrogram_source(:,:,2))))

% figure(4)
% imagesc(db(abs(spectrogram_source(:,:,3))))
% 
% figure(5)
% imagesc(db(abs(spectrogram_source(:,:,4))))

source_est = [];
for z = 1:Z
    S = spectrogram_source(:,:,z);
    out(1:Nshift*(T-1)+N_window,1)=0;
    
    for t=1:T
        Sfft=[S(:,t);conj(S(Nfft/2:-1:2,t))];
        stemp=real(ifft(Sfft));
        sout=stemp(1:N_window,1) .* sqrt(Window);
        
        out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)=out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)+sout;
    end
    
    source_est(z,:) = out;% ./ max(abs(out));
    audiowrite(['Results/',type,'/source_',num2str(z),'.wav'],source_est(z,:),fs)
end

% N_1 = size(source,2);
% N_2 = size(source_est,2);
% if N_1 < N_2
%     source_est = source_est(:,1:N_1);
% elseif N_2 < N_1
%     source = source(:,1:N_2);
% end

%[SDR,SIR,SAR,perm] = bss_eval_sources(source_est,source);
% iter_max = 50 : SDR = [6.2567; 9.7341];

%% Real-time PLCA
%class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};
%load('feature_norm.mat')
load('raw_norm_separation.mat')

idx_class_1 = find(raw_norm(:,end-1) == 1);
idx_class_2 = find(raw_norm(:,end-1) == 2);
aux_L = raw_norm(union(idx_class_1,idx_class_2),:);
aux_L(:,1:end-2) = aux_L(:,1:end-2).^2 ./ repmat(sum(aux_L(:,1:end-2).^2,2),[1,F]);
aux_L(:,1:end-2) = log(aux_L(:,1:end-2));
prior_g = [0.5,0.5];
my_class = [1,2];
param.N_spect = F;
param.gpuFlag = 0;
param.n_buff = 1;
param.verbose = 0;
param.dict = 1;

P_f = [mean(raw_norm(idx_class_1,1:F));
       mean(raw_norm(idx_class_2,1:F))];

Z = 2;
R = 1;
iter_max = 100;
fixed = {[],P_f,[]};
L_bay = zeros(T,Z);
mask = zeros(F,R,Z);
spectrogram_source = zeros(F,T,Z);
tic
for n = 1:T
    if n >= R
        [P_z,~,P_t] = plca(V(:,(n-R+1):n),Z,iter_max,fixed);
        for z = 1:Z
            mask(:,n,z) = P_z(z) * P_f(z,:)' * P_t(z,R);
        end
        mask(:,n,:) = mask(:,n,:) ./ repmat(sum(mask(:,n,:),3),[1,1,Z]);
        for z = 1:Z
            spectrogram_source(:,n,z) = spectrogram_complex(:,n) .* mask(:,n,z);
        end
        spectrums = reshape(abs(spectrogram_source(:,n,:)).^2,[F,Z])';
        spectrums = F .* spectrums(:,1:F) ./ repmat(sum(spectrums(:,1:F),2),[1,F]);
        spectrums = [spectrums, ones(2)];
        [~,posterior_g] = identification_general(spectrums, aux_L, prior_g, my_class, param);
        [~,L_bay(n,:)] = max(posterior_g);
        if L_bay(n,1) > L_bay(n,2)
             aux = spectrogram_source(:,n,1);
             spectrogram_source(:,n,1) = spectrogram_source(:,n,2);
             spectrogram_source(:,n,2) = aux;
             aux = L_bay(n,1);
             L_bay(n,1) = L_bay(n,2);
             L_bay(n,2) = L_bay(n,1);
        end
    else
        for z = 1:Z
            mask(:,n,z) = 1;
            spectrogram_source(:,n,z) = spectrogram_complex(:,n) .* mask(:,n,z);
        end
    end
end
elapsed_time = toc / T
% mask = mask ./ repmat(sum(mask,3),[1,1,Z]);
% % Recover sources
% spectrogram_source = zeros(F,T,Z);
% for z = 1:Z
%     spectrogram_source(:,:,z) = spectrogram_complex .* mask(:,:,z);
% end

figure(2)
imagesc(db(abs(spectrogram_source(:,:,1))))

figure(3)
imagesc(db(abs(spectrogram_source(:,:,2))))

source_est = [];
for z = 1:Z
    S = spectrogram_source(:,:,z);
    out(1:Nshift*(T-1)+N_window,1)=0;
    
    for t=1:T
        Sfft=[S(:,t);conj(S(Nfft/2:-1:2,t))];
        stemp=real(ifft(Sfft));
        sout=stemp(1:N_window,1) .* sqrt(Window);
        
        out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)=out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)+sout;
    end
    
    source_est(z,:) = out;
    audiowrite(['Results/',type,'/real-time/source_',num2str(z),'.wav'],source_est(z,:),fs)
end

N_1 = size(source,2);
N_2 = size(source_est,2);
if N_1 < N_2
    source_est = source_est(:,1:N_1);
elseif N_2 < N_1
    source = source(:,1:N_2);
end

[SDR,SIR,SAR,perm] = bss_eval_sources(source_est,source);
% iter_max = 100 : SDR = [3.11; 6.83];

%% Extend LVF to our dictionary based method
file = load('raw_norm.mat');
raw_norm = file.raw_norm;
class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};

true_class = [2,3];
n_class = length(class);
n_models = zeros(1,n_class);
for k = 1:n_class
    n_models(k) = sum(raw_norm(:,end-1) == k);
end

S = 2; % Number of sources
iter_max = 50;
P_f = cell(1,S);
for s = 1:S
    P_f{s} = abs(raw_norm(raw_norm(:,end-1) == true_class(s),1:N_window/2+1));
    P_f{s} = P_f{s} ./ repmat(sum(P_f{s},2),[1,N_window/2+1]);
end

[P_s,P_t] = separation_plca(V,P_f,iter_max);

model = cell(1,S);
for s = 1:S
    model{s} = reshape(sum(repmat(P_t{s},[1,F,1]) .* repmat(P_f{s},[1,1,T])),[F,T]);
end

mask = zeros(F,T,S);
for s = 1:S
    mask(:,:,s) = repmat(reshape(P_s(s,:,:),[1,T]),[F,1]) .* model{s};
end
mask = mask ./ repmat(sum(mask,3),[1,1,S]);

% Recover sources
spectrogram_source = zeros(F,T,S);
for s = 1:S
    spectrogram_source(:,:,s) = ((V./factor) .* mask(:,:,s)) .* exp(1i * phase_spect);
end

figure(2)
imagesc(db(abs(spectrogram_source(:,:,1))))

figure(3)
imagesc(db(abs(spectrogram_source(:,:,2))))

source_est = [];
for s = 1:S
    Spect = spectrogram_source(:,:,s);
    out(1:Nshift*(T-1)+N_window,1)=0;
    
    for t=1:T
        Sfft=[Spect(:,t);conj(Spect(Nfft/2:-1:2,t))];
        stemp=real(ifft(Sfft));
        sout=stemp(1:N_window,1) .* Window;
        
        out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)=out(Nshift*(t-1)+1:Nshift*(t-1)+N_window,1)+sout;
    end
    
    source_est(s,:) = out;
    audiowrite(['Results/mymethod/source_',num2str(s),'.wav'],source_est(s,:),fs)
end

N_1 = size(source,2);
N_2 = size(source_est,2);
if N_1 < N_2
    source_est = source_est(:,1:N_1);
elseif N_2 < N_1
    source = source(:,1:N_2);
end

[SDR,SIR,SAR,perm] = bss_eval_sources(source_est,source);
% iter_max = 100 : SDR = [3.11; 6.83];