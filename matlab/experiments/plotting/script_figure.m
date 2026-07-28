T = 1024;
D = 512;

[data,fs] = audioread('F:\Ph. D Thesis\Data\voice_female\voice_female_004.wav');

x = data(60850:60850+3*T);
t = 0:1/fs:(length(x)-1)/fs;

figure(100)
clf
plot(t,x)
xlabel('Time (s)')
ylabel('Amplitude')
axis([0 max(t) -1 1])
hold on
i = 2;
plot([t((i-1)*D+1),t((i-1)*D+1),t((i-1)*D+T),t((i-1)*D+T),t((i-1)*D+1)],[-1,1,1,-1,-1],'r-', 'LineWidth',2)
i = i + 1;
plot([t((i-1)*D+1),t((i-1)*D+1),t((i-1)*D+T),t((i-1)*D+T),t((i-1)*D+1)],[-1,1,1,-1,-1],'g--', 'LineWidth',2)
i = i + 1;
plot([t((i-1)*D+1),t((i-1)*D+1),t((i-1)*D+T),t((i-1)*D+T),t((i-1)*D+1)],[-1,1,1,-1,-1],'r-', 'LineWidth',2)

f = 0:fs/T:(T-1)*fs/T;
B = x(1:T);
S = fft(B);
S_energy = abs(S).^2;
N_spect = 205;
S_energy = N_spect*S_energy(1:N_spect) / sum(S_energy(1:N_spect));

figure(101)
clf
n_bins = 100;
stem(f(1:n_bins),S_energy(1:n_bins))
xlabel('Frequency (Hz)')
ylabel('Amplitude')





S_complex = [real(S(1:N_spect)),imag(S(1:N_spect))];
S_complex = N_spect * S_complex ./ repmat(sum(abs(S_complex).^2),[N_spect,1]);
S_complex = [f(1:N_spect)',S_complex];
S_complex = [real(S(1:N_spect)),imag(S(1:N_spect))];
S_complex = N_spect * S_complex ./ repmat(sum(abs(S_complex).^2),[N_spect,1]);
f(1:N_spect)',
S_complex
S_complex = [f(1:N_spect)',S_complex];
figure(3)
clf
plot3(S_complex(:,1),S_complex(:,2),S_complex(:,3) ,'*')
xlabel('Frequency (Hz)')
ylabel('Real(Spectrum)')
zlabel('Imag(Spectrum)')
clear
clc
x = -2:0.1:2;
relu = max(0,x);
plot(relu)
plot(x,relu)
axis([-2,2,-2,2])
figure(1)
plot([0,0],[-2,2],'--')
hold on
plot([-2,2],[0,0],'--')
figure(1)
clf
plot([0,0],[-2,2],'--k')
hold on
plot([-2,2],[0,0],'--k')