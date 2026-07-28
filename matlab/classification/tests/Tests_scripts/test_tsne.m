clear
clc
data_folder = '../../Data/';
folder_esc = 'ESC-10/';
addpath(genpath('AudioDescriptors'))
addpath(genpath('Statistics'))
addpath(genpath('Features'))
addpath(genpath('Results'))
addpath(genpath('tSNE_matlab'))
warning off

%% Initialisation
fs = 44100;                    % Sampling rate
D_s = 0.01;                    % Time interval between two analysys windows (sec)

T = [512, 1024, 2048, 4096];
D = round(2^nextpow2(D_s*fs)); % Time interval between two analysys windows (sample)

N_fft = T;                     % FFT size of the analysis window
N_spect = round(N_fft/5);      % Number of points kept in the spectrum

my_names = dir([data_folder,folder_esc]);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end

%class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
n_class = length(class);

folder_feat = 'Clean database features/';
folder_interval = 'Interval/';
type = '';

%%
dico = 3;

load(['Features/',folder_esc,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_T',num2str(T(dico)),'.mat'])

[n,p] = size(aux_L);
clear aux_L

freq = 0:fs/T(1):fs/2;
f = freq(1:N_spect(1)+1);

mypdf = zeros(n,N_spect(1)+2);
cnt = 1;
for k = 1:n_class
    disp(class{k})
    load(['Features/',folder_esc,folder_interval,'T',num2str(T(dico)),'/feature_',class{k},'.mat'])
    for i = 1:length(feature)
        for j = 1:length(feature{i})
            if ~isempty(feature{i}{j}.spectrum_model)
                mypdf(cnt,:) = [model2pdf(feature{i}{j}.spectrum_model,f,0),k];
                cnt = cnt + 1;
            end
        end
        progressbar(i,length(feature))
    end
end

%%
percent_test = 0.06;
mypdf_test = mypdf(sort(randperm(size(mypdf,1), round(size(mypdf,1)*percent_test))),:);
ydata = tsne(sqrt(mypdf_test(:,1:end-1)), mypdf_test(:,end));