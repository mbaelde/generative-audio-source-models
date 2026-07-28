clear
close all

addpath(genpath('Toolbox'))

[x1,fs] = audioread('F:\Ph. D Thesis\Data\A-Volute\detonation\gunshot_004.wav');
[x2,fs] = audioread('F:\Ph. D Thesis\Data\A-Volute\voice\voice_male_005.wav');

type = 1; %1 is temporal, fs is sample

x = [x1;zeros(round(fs/16),1);x2];
t = (0:1/fs:(length(x)-1)/fs)'*type;

framelength = 2048/fs*type;
shiftlength = 512/fs*type;

idx_11 = 2*shiftlength;
idx_12 = idx_11 + framelength;

idx_21 = 90*shiftlength;
idx_22 = idx_21 + framelength;

idx_31 = 180*shiftlength;
idx_32 = idx_31 + framelength;

t_1 = t(round(idx_11*(fs/type)+1:idx_12*(fs/type)));
x_1 = x(round(idx_11*(fs/type)+1:idx_12*(fs/type)));

t_2 = t(round(idx_21*(fs/type)+1:idx_22*(fs/type)));
x_2 = x(round(idx_21*(fs/type)+1:idx_22*(fs/type)));

t_3 = t(round(idx_31*(fs/type)+1:idx_32*(fs/type)));
x_3 = x(round(idx_31*(fs/type)+1:idx_32*(fs/type)));

N_fft = 2048;
if type == 1
    freq = 0:fs/N_fft:(N_fft/2)*fs/N_fft;
elseif type == fs
    freq = 1:N_fft/2+1;
end
X_1 = abs(fft(x_1)).^2;
X_1 = X_1(1:N_fft/2+1) / sum(X_1(1:N_fft/2+1));

X_2 = abs(fft(x_2)).^2;
X_2 = X_2(1:N_fft/2+1) / sum(X_2(1:N_fft/2+1));

X_3 = abs(fft(x_3)).^2;
X_3 = X_3(1:N_fft/2+1) / sum(X_3(1:N_fft/2+1));

myred = [217,83,25]./256;
h = figure('DefaultAxesFontSize',12);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 560, 260])
plot(t,x)
hold on
plot([idx_11,idx_12,idx_12,idx_11,idx_11],[-1,-1,1,1,-1],'Color',myred,'LineWidth',2)
%plot([idx_21,idx_22,idx_22,idx_21,idx_21],[-1,-1,1,1,-1],'Color',myred,'LineWidth',2)
%plot([idx_31,idx_32,idx_32,idx_31,idx_31],[-1,-1,1,1,-1],'Color',myred,'LineWidth',2)
if type == 1
    xlabel('Time (s)')
elseif type == fs
    xlabel('Sample')
end
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
set(gcf, 'Position', [1350, 500, 280, 200])
plot(t_1,x_1)
if type == 1
    xlabel('Time (s)')
elseif type == fs
    xlabel('Sample')
end
ylabel('Amplitude')
xlim([t_1(1),t_1(end)])
ylim([-1,1])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'frame_1' )
movefile('frame_1.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\frame_1.pdf')
movefile('frame_1.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\frame_1.pdf_tex')

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 280, 200])
plot(t_2,x_2)
if type == 1
    xlabel('Time (s)')
elseif type == fs
    xlabel('Sample')
end
ylabel('Amplitude')
xlim([t_2(1),t_2(end)])
ylim([-0.05,0.05])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'frame_2' )
movefile('frame_2.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\frame_2.pdf')
movefile('frame_2.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\frame_2.pdf_tex')

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 280, 200])
plot(t_3,x_3)
if type == 1
    xlabel('Time (s)')
elseif type == fs
    xlabel('Sample')
end
ylabel('Amplitude')
xlim([t_3(1),t_3(end)])
ylim([-0.4,0.4])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'frame_3' )
movefile('frame_3.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\frame_3.pdf')
movefile('frame_3.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\frame_3.pdf_tex')

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 280, 200])
plot(freq,X_1)
if type == 1
    xlabel('Frequency (kHz)')
elseif type == fs
    xlabel('Frequency bins')
end
ylabel('Power')
xlim([freq(1),freq(end)])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'spectrum_1' )
movefile('spectrum_1.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrum_1.pdf')
movefile('spectrum_1.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrum_1.pdf_tex')

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 280, 200])
plot(freq,X_2)
if type == 1
    xlabel('Frequency (kHz)')
elseif type == fs
    xlabel('Frequency bins')
end
ylabel('Power')
xlim([freq(1),freq(end)])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'spectrum_2' )
movefile('spectrum_2.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrum_2.pdf')
movefile('spectrum_2.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrum_2.pdf_tex')

h = figure('DefaultAxesFontSize',10);
clf
set(gca,'DefaultTextFontname', 'CMU Serif')
set(gca,'DefaultAxesFontName', 'CMU Serif')
set(gcf, 'Position', [1350, 500, 280, 200])
plot(freq,X_3)
if type == 1
    xlabel('Frequency (kHz)')
elseif type == fs
    xlabel('Frequency bins')
end
ylabel('Power')
xlim([freq(1),freq(end)])
ChangeInterpreter(gcf,'Latex');
Plot2LaTeX( h, 'spectrum_3' )
movefile('spectrum_3.pdf', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrum_3.pdf')
movefile('spectrum_3.pdf_tex', 'F:\Ph. D Thesis\Softwares\Toolbox\Plot2LaTex_v1d2\Example\spectrum_3.pdf_tex')

