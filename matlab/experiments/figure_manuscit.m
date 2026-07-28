clear
close all

addpath(genpath('Toolbox'))

[x1,fs] = audioread('F:\Ph. D Thesis\Data\A-Volute\voice\voice_male_005.wav');

x = x1;
t = (0:1/fs:(length(x)-1)/fs)';

N_fft = length(x);

freq = 0:fs/N_fft:(N_fft/2)*fs/N_fft;

X = fft(x);
X = abs(X(1:N_fft/2+1));

h = figure('DefaultAxesFontSize',12);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 560, 260])
plot(t,x)
xlabel('Temps (s)')
ylabel('Amplitude')
xlim([0,t(end)])
ylim([-1,1])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'sounds' )
movefile('sounds.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\sounds.pdf')
movefile('sounds.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\sounds.pdf_tex')

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 560, 260])
plot(freq,X)
xlabel('Frequence (kHz)')
ylabel('Magnitude')
xlim([freq(1),freq(end)])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'spectrum_1' )
movefile('spectrum_1.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrum_1.pdf')
movefile('spectrum_1.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrum_1.pdf_tex')

n_window = 1024;
window = sqrt(hanning(n_window));
n_overlap = 512;
n_fft = 1024;
[spectrogram_x,freq_s,t_s] = spectrogram(x,window,n_overlap,n_fft);
%%
h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
imagesc(db(abs(spectrogram_x)))
set(gca,'YDir','normal')
xlabel('Temps (s)')
ylabel('Frequence (kHz)')
xt = xticks;
yt = yticks;
set(gca,'XTickLabel',round(100 * xt * n_overlap / fs) / 100)
set(gca,'YTickLabel',round(yt * fs / n_fft / 10) / 100)
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'spectrogram' )
movefile('spectrogram.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrogram.pdf')
movefile('spectrogram.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrogram.pdf_tex')

%%
t_wrec = linspace(-n_window,n_window,2*n_window + 1)/fs;
w_rec = [zeros(1,n_window/2),ones(1,n_window+1),zeros(1,n_window/2)];
t_w = linspace(-n_window/2,n_window/2,n_window + 1)/fs;
w_hanning = hanning(n_window+1)';
w_hamming =  hamming(n_window+1)';

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 260, 260])
plot(t_wrec,w_rec)
xlabel('Temps (s)')
ylabel('Amplitude')
xlim([t_wrec(1),t_wrec(end)])
ylim([-0.2,1.2])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'rectangulaire' )
movefile('rectangulaire.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\rectangulaire.pdf')
movefile('rectangulaire.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\rectangulaire.pdf_tex')

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 260, 260])
plot(t_w,w_hanning)
xlabel('Temps (s)')
ylabel('Amplitude')
xlim([t_w(1),t_w(end)])
ylim([-0.2,1.2])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'hanning' )
movefile('hanning.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\hanning.pdf')
movefile('hanning.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\hanning.pdf_tex')

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 260, 260])
plot(t_w,w_hamming)
xlabel('Temps (s)')
ylabel('Amplitude')
xlim([t_w(1),t_w(end)])
ylim([-0.2,1.2])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'hamming' )
movefile('hamming.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\hamming.pdf')
movefile('hamming.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\hamming.pdf_tex')
%%
n_window = 64;
window = sqrt(hanning(n_window));
n_overlap = n_window / 2 ;
n_fft = n_window;
[spectrogram_x,freq_s,t_s] = spectrogram(x,window,n_overlap,n_fft);

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
imagesc(db(abs(spectrogram_x)))
set(gca,'YDir','normal')
xlabel('Temps (s)')
ylabel('Frequence (kHz)')
xt = xticks;
yt = yticks;
set(gca,'XTickLabel',round(100 * xt * n_overlap / fs) / 100)
set(gca,'YTickLabel',round(yt * fs / n_fft / 10) / 100)
ChangeInterpreter(gcf,'Latex');
