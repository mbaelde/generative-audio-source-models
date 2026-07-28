clear
clc
% addpath(genpath('Toolbox'))
% addpath(genpath('Prototypes/Separation'))
addpath(genpath(pwd))
%distcomp.feature( 'LocalUseMpiexec', false )
%% Load data
folder_mixture = '\\AUDIO06\ServeurMatlab\Ph. D Thesis\Data\MUS2016\DSD100\Mixtures\Test\';
folder_source = '\\AUDIO06\ServeurMatlab\Ph. D Thesis\Data\MUS2016\DSD100\Sources\Test\';

names = dir(folder_mixture);
names = names(3:end);
n_band = length(names);

%% PLCS
P_f = [];
Z = [];
n_clusters = [30,80,30,30,30];
file_1 = load(['Prototypes/Separation/Data/dict_drums_acoustic.mat']);
file_2 = load(['Prototypes/Separation/Data/dict_drums_electro.mat']);
V = [file_1.V;file_2.V];
%V = file_1.V;
[~,C] = kmedoids(V,n_clusters(1));
P_f = [P_f; C];
Z(1) = size(C,1);
file_1 = load(['Prototypes/Separation/Data/dict_vocals_acoustic.mat']);
file_2 = load(['Prototypes/Separation/Data/dict_vocals_metal.mat']);
V = [file_1.V;file_2.V];
%V = file_1.V;
[~,C] = kmedoids(V,n_clusters(2));
P_f = [P_f; C];
Z(2) = size(C,1);
file_1 = load(['Prototypes/Separation/Data/dict_bass_acoustic.mat']);
file_2 = load(['Prototypes/Separation/Data/dict_bass_electro.mat']);
V = [file_1.V;file_2.V];
%V = file_1.V;
[~,C] = kmedoids(V,n_clusters(3));
P_f = [P_f; C];
Z(3) = size(C,1);
file_1 = load(['Prototypes/Separation/Data/dict_guitar_acoustic.mat']);
file_2 = load(['Prototypes/Separation/Data/dict_guitar_metal.mat']);
V = [file_1.V;file_2.V];
%V = file_1.V;
[~,C] = kmedoids(V,n_clusters(4));
P_f = [P_f; C];
Z(4) = size(C,1);
file_1 = load(['Prototypes/Separation/Data/dict_instrumental_acoustic.mat']);
file_2 = load(['Prototypes/Separation/Data/dict_instrumental_electro.mat']);
V = [file_1.V;file_2.V];
%V = file_1.V;
[~,C] = kmedoids(V,n_clusters(5));
P_f = [P_f; C];
Z(5) = size(C,1);
%% PLCS 2 class
P_f = [];
Z = [];
n_clusters = [40,60];
file_1 = load(['Prototypes/Separation/Data/dict_drums_acoustic.mat']);
file_2 = load(['Prototypes/Separation/Data/dict_drums_electro.mat']);
file_3 = load(['Prototypes/Separation/Data/dict_bass_acoustic.mat']);
file_4 = load(['Prototypes/Separation/Data/dict_bass_electro.mat']);
file_5 = load(['Prototypes/Separation/Data/dict_guitar_acoustic.mat']);
file_6 = load(['Prototypes/Separation/Data/dict_guitar_metal.mat']);
file_7 = load(['Prototypes/Separation/Data/dict_instrumental_acoustic.mat']);
file_8 = load(['Prototypes/Separation/Data/dict_instrumental_electro.mat']);
V = [file_1.V;file_2.V;file_3.V;file_4.V;file_5.V;file_6.V;file_7.V;file_8.V];
%V = file_1.V;
[~,C] = kmedoids(V,n_clusters(1));
P_f = [P_f; C];
Z(1) = size(C,1);
file_1 = load(['Prototypes/Separation/Data/dict_vocals_acoustic.mat']);
%file_2 = load(['Prototypes/Separation/Data/dict_vocals_metal.mat']);
%V = [file_1.V;file_2.V];
V = file_1.V;
[~,C] = kmedoids(V,n_clusters(2));
P_f = [P_f; C];
Z(2) = size(C,1);
% file_1 = load(['Prototypes/Separation/Data/dict_instrumental_acoustic.mat']);
% file_2 = load(['Prototypes/Separation/Data/dict_instrumental_electro.mat']);
% %V = [file_1.V;file_2.V];
% V = file_1.V;
% [~,C] = kmedoids(V,n_clusters(5));
% P_f = [P_f; C];
% Z(5) = size(C,1);
%%
for n = 1:n_band
    disp(['Process ',names(n).name,'...'])
[mixture,fs] = audioread([folder_mixture,names(n).name,'\mixture.wav']);
source_1 = audioread([folder_source,names(n).name,'\bass.wav']);
source_2 = audioread([folder_source,names(n).name,'\drums.wav']);
source_3 = audioread([folder_source,names(n).name,'\other.wav']);
source_4 = audioread([folder_source,names(n).name,'\vocals.wav']);

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

start_time = 70;
end_time = 80;

source{1} = source_1(:,start_time*fs:end_time*fs);
source{2} = source_2(:,start_time*fs:end_time*fs);
source{3} = source_3(:,start_time*fs:end_time*fs);
source{4} = source_4(:,start_time*fs:end_time*fs);

mixture = mixture(:,start_time*fs:end_time*fs);
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
classes = {'Drums','Vocals','Bass','Guitar','Instrumental'};
%classes = {'Rest','Vocals'};
sources_est_plcs = cell(1,n_source);
if ~exist(['Prototypes/Separation/Results/Benchmark/',names(n).name,'/'])
    mkdir(['Prototypes/Separation/Results/Benchmark/',names(n).name,'/']);
end
for s = 1:n_source
    spectrogram_dict = rec_spectrogram(:,:,:,s) .* exp(1i * phase_spect);
    
    sources_est_plcs{s}(1:Nshift*(T-1)+N_window,1)=0;
    sources_est_plcs{s}(1:Nshift*(T-1)+N_window,2)=0;
    for i = 1:I
        S = spectrogram_dict(:,:,i);

        for t = 1:T
            Sfft = [S(:,t);conj(S(N_window/2:-1:2,t))];
            stemp = real(ifft(Sfft));
            sout = stemp(1:N_window,1) .* sqrt(Window);

            sources_est_plcs{s}(Nshift*(t-1)+1:Nshift*(t-1)+N_window,i) = sources_est_plcs{s}(Nshift*(t-1)+1:Nshift*(t-1)+N_window,i)+sout;
        end
    end

    audiowrite(['Prototypes/Separation/Results/Benchmark/',names(n).name,'/PLCS_',classes{s},'_reconstruct.wav'],sources_est_plcs{s},fs)
end

%% Extraction de voix centrale
[sources_est_evc,Nombre_Fenetre] = Extraction_Voix_Centrale(mixture',fs,'Temporel');

for s = 1:3
    audiowrite(['Prototypes/Separation/Results/Benchmark/',names(n).name,'/EVC_source_',num2str(s),'_reconstruct.wav'],sources_est_evc(:,s),fs)
end
% end
%%
for n = 1:n_band
    [mixture,fs] = audioread([folder_mixture,names(n).name,'\mixture.wav']);
source_1 = audioread([folder_source,names(n).name,'\bass.wav']);
source_2 = audioread([folder_source,names(n).name,'\drums.wav']);
source_3 = audioread([folder_source,names(n).name,'\other.wav']);
source_4 = audioread([folder_source,names(n).name,'\vocals.wav']);

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

start_time = 70;
end_time = 80;

source{1} = source_1(:,start_time*fs:end_time*fs);
source{2} = source_2(:,start_time*fs:end_time*fs);
source{3} = source_3(:,start_time*fs:end_time*fs);
source{4} = source_4(:,start_time*fs:end_time*fs);

mixture = mixture(:,start_time*fs:end_time*fs);

    for s = 1:2
        aux = audioread(['Prototypes/Separation/Results/Benchmark voice-rest/',names(n).name,'/PLCS_',classes{s},'_reconstruct.wav']);
        sources_est_plcs{s} = aux;
    end
    for s = 1:3
        aux = audioread(['Prototypes/Separation/Results/Benchmark voice-rest/',names(n).name,'/EVC_source_',num2str(s),'_reconstruct.wav']);
        sources_est_evc(:,s) = aux;
    end
% Comparaison
sources_est_plcs_mono = [];
source_mono = [];
for s = 1:n_source
    sources_est_plcs_mono(s,:) = mean(sources_est_plcs{s},2);
end
for s = 1:4
    source_mono(s,:) = mean(source{s});
end

N = size(sources_est_plcs_mono,2);
% % Metric only to voice for plcs
% [SDR,SIR,SAR,perm] = bss_eval_sources(sources_est_plcs_mono(2,:),mean(source_2(:,1:N)));
% % Metric for all sources for plcs
% [SDR,SIR,SAR,perm] = bss_eval_sources(sources_est_plcs_mono, source_mono(:,1:N));
% Metric for all voice/rest for evc
sources_plcs(1,:) = mean(sources_est_plcs_mono([1,3,4,5],:));
sources_plcs(2,:) = sources_est_plcs_mono(2,:);
%sources_plcs = sources_est_plcs_mono;
sources_t(1,:) = mean(source_mono([1,3,4],:));
sources_t(2,:) = source_mono(2,:);
[SDR_plcs(:,n),SIR_plcs(:,n),SAR_plcs(:,n),~] = bss_eval_sources(sources_plcs, sources_t(:,1:N));

% % Metric only to voice for evc
% [SDR,SIR,SAR,perm] = bss_eval_sources(sources_est_evc(1:N,3)',mean(source_2));
% Metric for voice/side for evc
sources_evc(1,:) = mean(sources_est_evc(:,1:2)');
sources_evc(2,:) = sources_est_evc(:,3)';
sources_t(1,:) = mean(source_mono([1,3,4],:));
sources_t(2,:) = source_mono(2,:);
N = size(sources_t,2);
[SDR_evc(:,n),SIR_evc(:,n),SAR_evc(:,n),~] = bss_eval_sources(sources_evc(:,1:N), sources_t);

% clc
% fileID = fopen(['Prototypes/Separation/Results/Benchmark/',names(n).name,'/results.txt'],'w');
% fprintf(fileID,'--------------------------------\n');
% fprintf(fileID,'\nSDR :        PLCS  -  EVC\n');
% fprintf(fileID,'\nRest/Side:   %2.2f    %2.2f\n',SDR_plcs(1), SDR_evc(1));
% fprintf(fileID,'\nVoice:       %2.2f    %2.2f\n',SDR_plcs(2), SDR_evc(2));
% fprintf(fileID,'\n--------------------------------\n');
% fprintf(fileID,'\nSAR :        PLCS  -  EVC\n');
% fprintf(fileID,'\nRest/Side:   %2.2f    %2.2f\n',SAR_plcs(1), SAR_evc(1));
% fprintf(fileID,'\nVoice:       %2.2f    %2.2f\n',SAR_plcs(2), SAR_evc(2));
% fprintf(fileID,'\n--------------------------------\n');
% fprintf(fileID,'\nSIR :        PLCS  -  EVC\n');
% fprintf(fileID,'\nRest/Side:   %2.2f    %2.2f\n',SIR_plcs(1), SIR_evc(1));
% fprintf(fileID,'\nVoice:       %2.2f    %2.2f\n',SIR_plcs(2), SIR_evc(2));
% fprintf(fileID,'\n--------------------------------\n');
% fclose(fileID);
end