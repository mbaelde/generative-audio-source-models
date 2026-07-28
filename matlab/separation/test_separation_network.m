clear
clc
addpath(genpath(pwd))

%% Load data
folder_mixture = '\\AUDIO06\ServeurMatlab\Ph. D Thesis\Data\MUS2016\DSD100\Mixtures\Test\';
folder_source = '\\AUDIO06\ServeurMatlab\Ph. D Thesis\Data\MUS2016\DSD100\Sources\Test\';

folder_mixture = 'F:\Ph. D Thesis\Data\MUS2016\DSD100\Mixtures\Test\';
folder_source = 'F:\Ph. D Thesis\Data\MUS2016\DSD100\Sources\Test\';

names = dir(folder_mixture);
names = names(3:end);
n_band = length(names);

%%
X = cell(1);
S = cell(1);
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

% Compute spectrogram
n_source = 4;
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


%% Feed-Forward Neural Network
% Compute non overlapping spectrogram
n_source = 4;
N_window = 1024;
Window = hanning(N_window);
Nshift = N_window/2;
Nfft = N_window;

spectrogram_mixture = spectrogram(sum(mixture,1),sqrt(Window),Nshift,Nfft);
V_mixture = abs(spectrogram_mixture);
phase_spect = angle(spectrogram_mixture);

[F,T] = size(V_mixture);
% Compute feature vector
C = 2; % context window
m = C+1; % start index
X = V_mixture(:,m-C:m+C); % feature vector
X = X(:);
gamma_norm = sum(abs(X).^2); % normalization
X = X ./ gamma_norm;

% Construct network
layers = [
    imageInputLayer([(2*C+1)*F 1 1]);
    
    fullyConnectedLayer(F)
    reluLayer
    
    fullyConnectedLayer(F)
    reluLayer
    
    fullyConnectedLayer(F)
    reluLayer
    
    fullyConnectedLayer(F)
    reluLayer
    
    fullyConnectedLayer(F)
    reluLayer];

%%
start_time = 70;
end_time = 80;

source{1} = source_1(:,start_time*fs:end_time*fs);
source{2} = source_2(:,start_time*fs:end_time*fs);
source{3} = source_3(:,start_time*fs:end_time*fs);
source{4} = source_4(:,start_time*fs:end_time*fs);

mixture = mixture(:,start_time*fs:end_time*fs);

%% Multichannel Wiener Filtering
rec_spectrogram = zeros(F,T,I,n_source);
v = rand(F,T,n_source);
R = rand(F,I,I,n_source);

iter_max = 100;

for iter = 1:iter_max
    % Update source spectrogram
    parfor s = 1:n_source
        for f = 1:F
            for t = 1:T
                rec_spectrogram(f,t,:,s) = v(f,t,s) .* squeeze(R(f,:,:,s)) * ((squeeze(sum(v(f,t,s) .* R(f,:,:,:),4))) \ squeeze(V(f,t,:)));
            end
        end
    end
    
    % Update PSD
    v = squeeze(sum(abs(rec_spectrogram).^2,3));
    
    % Update Spatial Covariance
    parfor s = 1:n_source
        for f = 1:F
            aux = 0;
            for n = 1:T
                aux = squeeze(rec_spectrogram(f,n,:,s)) * squeeze(rec_spectrogram(f,n,:,s))';
            end
            R(f,:,:,s) = aux ./ sum(v(f,:,s),2);
        end
    end
    disp(['iter: ',num2str(iter),' / ',num2str(iter_max)])
end
% Reconstruct sources
parfor s = 1:n_source
    for f = 1:F
        for t = 1:T
            rec_spectrogram(f,t,:,s) = v(f,t,s) .* squeeze(R(f,:,:,s)) * ((squeeze(sum(v(f,t,s) .* R(f,:,:,:),4))) \ squeeze(V(f,t,:)));
        end
    end
end

sources_est_plcs = cell(1,n_source);
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

    audiowrite(['MWF_source',num2str(s),'_reconstruct.wav'],sources_est_plcs{s},fs)
end
