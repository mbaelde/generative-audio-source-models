clear
clc
data_folder = '../../Data/';
folder_database = 'A-Volute/';
addpath(genpath('AudioDescriptors'))
addpath(genpath('Statistics'))
addpath(genpath('Features'))
addpath(genpath('Results'))
addpath(genpath('Database'))
warning off

distcomp.feature( 'LocalUseMpiexec', false )
%%
dico = 2;

%% Define parameters for mixture models estimation of the spectrum
fs = 44100;                                         % Sampling rate
D_s = 0.01;                                         % Time interval between two analysys windows (sec)

T = 2.^(9:15);
D = round(2^nextpow2(D_s*fs));                      % Time interval between two analysys windows (sample)

N_fft = T;                                          % FFT size of the analysis window
N_spect = round(N_fft/5);                           % Number of points kept in the spectrum
criterion = 'BIC';                                  % Criterion used for choosing number of mixture components
verbose = 0;                                        % Flag to display information during estimation process
M_set = 1:20;                                       % Candidate values for the number of mixture components

param.T = T;
param.D = D;
param.N_spect = N_spect;
param.M_set = M_set;
param.criterion = criterion;
param.verbose = verbose;

%class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
my_names = dir([data_folder,folder_database]);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end
class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};
n_class = length(class);

%% Estimate dictionnaries
clear feature
disp(['Computing dictionnary for T = ', num2str(T(dico))])
param.T = T(dico);
param.N_spect = N_spect(dico);

load(['Database/',folder_database,'FS',num2str(fs),'/T',num2str(T(dico)),'/Uniform/database_overlap.mat'])

for k = 1:n_class
    idx{k} = find(database(:,end-1) == k);
end
aux_database = database;
aux_database(union(union(idx{1},idx{5}),idx{7}),end-1) = 1; % Engine
aux_database(union(idx{3},idx{4}),end-1) = 2; % Detonation
aux_database(union(idx{8},idx{9}),end-1) = 3; % Voice
aux_database(idx{2},end-1) = 4; % Alarm
aux_database(idx{6},end-1) = 5; % Step 
database = [];
for k = 1:5
    database = [database; aux_database(aux_database(:,end-1) == k,:)];
end
clear aux_database
save(['Database/',folder_database,'FS',num2str(fs),'/T',num2str(T(dico)),'/Metaclasse/database_overlap.mat'],'database')


feature = zeros(size(database,1),N_spect(dico)+2);
raw_spectrum = zeros(size(database,1),T(dico)+2);
for i = 1:size(database,1)
    raw_spectrum(i,:) = [fft(database(i,1:end-2)), database(i,end-1:end)];
    spectrum = abs(raw_spectrum(i,1:end-2)).^2;
    spectrum = N_spect(dico) * spectrum(1:N_spect(dico)) / sum(spectrum(1:N_spect(dico)));
    feature(i,:) = [spectrum, database(i,end-1:end)];
    progressbar(i,size(database,1))
end

save(['Features/',folder_database,'/Metaclasse/T',num2str(T(dico)),'/feature_T',num2str(T(dico)),'.mat'],'feature','-v7.3')
save(['Features/',folder_database,'FS',num2str(fs),'/T',num2str(T(dico)),'/Uniform/raw_spectrum_T',num2str(T(dico)),'.mat'],'raw_spectrum','-v7.3')
%%
f = 0:fs/T(dico):(T(dico)-1)*fs/T(dico);
delta = f(2)-f(1);
aux_L = feature;

aux_L(:,1:end-2) = log(aux_L(:,1:end-2));

% parfor i = 1:size(feature,1)
%     aux_L(i,:) = [log(feature(i,1:end-2)),feature(i,end-1:end)];
%     %progressbar(i,size(feature,1));
% end

save(['Features/',folder_database,'/Uniform/T',num2str(T(dico)),'/preprocess_dictionnary_T',num2str(T(dico)),'.mat'],'aux_L','-v7.3')
