sound_folder = ['../../Data/',folder_database]; % dataset lost, kept for reference only
addpath(genpath('common'))
addpath(genpath('classification'))
addpath(genpath('separation'))
addpath(genpath('experiments'))
warning off


%% Initialisation
fs = 44100;                    % Sampling rate
D_s = 0.01;                    % Time interval between two analysys windows (sec)

T = [512, 1024, 2048, 4096];
D = round(2^nextpow2(D_s*fs)); % Time interval between two analysys windows (sample)

fs = [11025,22050,44100];
N_fft = T;                     % FFT size of the analysis window
N_spect = round(N_fft/5);      % Number of points kept in the spectrum

%class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};

my_names = dir(sound_folder);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end

n_class = length(class);