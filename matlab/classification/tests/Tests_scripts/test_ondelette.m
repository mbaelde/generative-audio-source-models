clear
addpath(genpath('AudioDescriptors'))

%% Temporal signal
Fs = 44100;
N = 50000;

x = 0:1/Fs:(N-1)/Fs;
f = [1000, 5000];
y = 0.5*(sin(2*pi*x*f(1)) + sin(2*pi*x*f(2))) + 0.5*randn(1,N);

figure(1)
clf
plot(x(1:500),y(1:500))

%% STFT
winsize = 1024;
hopsize = winsize / 2;
n_fft = winsize;
stft_data = spectrogram(y, winsize, hopsize, n_fft);
f = 0:Fs/n_fft:Fs/2;

figure(2)
clf
subplot(2,1,1)
imagesc(abs(stft_data).^2);
set(gca,'ydir','normal')
subplot(2,1,2)
plot(f,abs(stft_data(:,1)).^2)

%% Wavelet
wname = 'coif3';
J = floor(log2(N));
[C,L] = wavedec(y,J,wname);
N_spect = N / 2;

spectrum = zeros(J,N_spect);
for j = 1:J
    if mod(N_spect/L(j+1),2) == 0
        spectrum(j,:) = reshape(repmat(abs(C(1+sum(L(1:j)):sum(L(1:j+1)))).^2, [N_spect/L(j+1),1]), [1,N_spect]);
    else
        delta = N_spect - L(j+1)*floor(N_spect/L(j+1));
        spectrum(j,:) = [reshape(repmat(abs(C(1+sum(L(1:j)):sum(L(1:j+1)))).^2, [floor(N_spect/L(j+1)),1]), [1,N_spect-delta]), zeros(1,delta)];
    end
end
% J = log2(winsize);
% N_buffer = floor((N - winsize) / hopsize);
%
% spectrum = zeros(J,N/2);
% for b = 1:N_buffer
%     [C,L] = wavedec(y(1+(b-1)*hopsize:(b-1)*hopsize+winsize),J,wname);
%
%     for j = 1:J
%         spectrum(j,1+(b-1)*hopsize:b*hopsize) = reshape(repmat(abs(C(1+sum(L(1:j)):sum(L(1:j+1)))).^2, [hopsize/L(j+1),1]), [1,hopsize]);
%     end
%     progressbar(b,N_buffer)
% end

[wavelet_data, TIMES, FREQ] = wavelet_spectrum(y, Fs, winsize, hopsize, n_fft);

figure(3)
clf
subplot(2,1,1)
imagesc(wavelet_data)
set(gca,'ydir','normal')
subplot(2,1,2)
plot(FREQ,wavelet_data(:,1))

%% Load real signal
data_folder = '../../Data/';
class = {'airplane', 'alarm', 'explosion', 'gunshot', 'helicopter', 'step', 'vehicule', 'voice_female', 'voice_male'};

wname = 'sym5';

n = 1;

folder = [data_folder,class{n}];
names = dir(folder);
names = names(3:end);
n_sounds(n) = length(names);

i = 1;

file = [folder,'/',names(i).name];
[sound,fs_sound] = audioread(file);

f_factor = Fs / fs_sound;
if f_factor ~= 1
    sound = resample(sound, f_factor, 1);
end
% Mean a stereo signal to form a mono signal
if size(sound,2) == 2
    sound = mean(sound,2);
end
sound = sound - mean(sound);

sound = sound(1:1024*4);

N_spect = n_fft/2;%floor(n_fft/5);

stft_data = abs(spectrogram(sound, winsize, hopsize, n_fft).^2);
stft_data = N_spect * stft_data(1:N_spect,:) ./ repmat(sum(stft_data(1:N_spect,:)),[N_spect,1]);

[wavelet_data, TIMES, FREQ] = wavelet_spectrum(sound, Fs, winsize, hopsize, n_fft, wname);
wavelet_data = N_spect * wavelet_data(1:N_spect,:) ./ repmat(sum(wavelet_data(1:N_spect,:)),[N_spect,1]);

[scalogram, ~, ~] = wavelet_scalogram(sound, wname);
%scalogram = size(scalogram,2) * scalogram ./ repmat(sum(scalogram,2),[1,size(scalogram,2)]);

scales = linspace(1.3333,682.666,512);
% [coeff,sgram] = cwt(sound,scales,wname,'scal');
% scalcoeff = 513 * abs(coeff).^2 ./ repmat(sum(abs(coeff).^2), [513,1]);
scalcoeff = continous_wavelet_scalogram(sound, scales, wname);

figure(4)
clf
subplot(5,1,1)
plot(sound)
title('Sound')
subplot(5,1,2)
imagesc(db(stft_data))
set(gca,'ydir','normal')
colormap jet
title('STFT')
subplot(5,1,3)
imagesc(db(wavelet_data))
set(gca,'ydir','normal')
colormap jet
title('Wavelet Spectrum')
subplot(5,1,4)
imagesc(db(scalogram))
%set(gca,'ydir','normal')
colormap jet
title('Discrete Wavelet Scalogram')
subplot(5,1,5)
imagesc(db(scalcoeff))
%set(gca,'ydir','normal')
colormap jet
title('Continuous Wavelet Scalogram')

% J = size(scalogram,1);
% [C,L] = wavedec(sound,J,wname);

% figure(5)
% clf
%subplot(5,2,1)
%plot(sound)
% title('Sound')
% for j = 1:J
%     subplot(5,2,j)
%     plot(1:L(j+1),C(1+sum(L(1:j)):sum(L(1:j+1))))
%     title(['Detailed level ',num2str(J-j+1)])
% end

% for b = 1:max(size(stft_data,2), size(wavelet_data,2))
% figure(5)
% clf
% subplot(2,1,1)
% plot(stft_data(:,b))
% subplot(2,1,2)
% plot(wavelet_data(:,b))
% pause(0.1)
% end

figure(100)
clf
for b = 1:size(scalcoeff,2)
    plot(scalcoeff(:,b))
    pause(0.01)
end

figure(200)
clf
for b = 1:size(scalogram,2)
    plot(scalogram(:,b))
    pause(0.01)
end
%%
N_sig = max(length(sound_gunshot), length(sound_step));
p = [0.1,0.9];

mixed = p(1) * sound_gunshot + p(2) * [sound_step; zeros(N_sig - length(sound_step),1)];
stft_mixed = abs(spectrogram(mixed, winsize, hopsize, n_fft).^2);
stft_mixed = stft_mixed(1:N_spect,:);

figure(10)
clf
plot(p(1) * sound_gunshot)
hold on
plot(p(2) * [sound_step; zeros(N_sig - length(sound_step),1)])
plot(mixed)
%%
b = 5;
figure(11)
clf
clf
plot(p(1) * N_spect * stft_gunshot(:,b) / sum(stft_gunshot(:,b)))
hold on
plot(p(2) * N_spect * stft_step(:,b) / sum(stft_step(:,b)))
plot(p(1) * N_spect * stft_gunshot(:,b) / sum(stft_gunshot(:,b)) + p(2) * N_spect * stft_step(:,b) / sum(stft_step(:,b)))
plot(N_spect * stft_mixed(:,b) / sum(stft_mixed(:,b)))
legend('Gunshot', 'Step', 'Summed','Mixed')