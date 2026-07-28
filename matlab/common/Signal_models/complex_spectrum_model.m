%% Script complex_spectrum_model
% Create a modelization of complex audio spectrum using GMM in 3D.
%
% Author: Maxime Baelde
% Date: April 2016 - Today
% Company: A-Volute / MODAL

clear
gcp
%% Initialisation
fs = 44100;                    % Sampling rate
D_s = 0.01;                    % Time interval between two analysys windows (sec)

T = [512, 1024, 2048];
D = round(2^nextpow2(D_s*fs)); % Time interval between two analysys windows (sample)

N_fft = T;                     % FFT size of the analysis window
N_spect = round(N_fft/5);      % Number of points kept in the spectrum

class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
n_class = length(class);
%%
% n_samp = 10;
% for k = 1:n_class
% load(['Database/T512/database_',class{k},'.mat'])
% n = size(database,1);
% idx = randperm(n,n_samp);
% reduced_database(1+(k-1)*n_samp:k*n_samp,:) = database(idx,:);
% end
dico = 2;
if dico == 1
    load(['Database/T',num2str(T(dico)),'/database.mat'])
else
    load(['Database/T',num2str(T(dico)),'/database_overlap.mat'])
end
N = size(database,1);

model_size = zeros(1,n_class);
for k = 1:n_class
    model_size(k) = sum(database(:,end-1) == k);
end
cum_model_size = [0,cumsum(model_size)];

M_set = 1:10;

f = 0:fs/N_fft(dico):fs/2;
f = f(1:N_spect(dico));
%%
for n = 1:n_class
    disp(['-- Class ', class{n},' --'])
    clear feature
    %n = 9;

    mydata = database(cum_model_size(n)+1:cum_model_size(n+1),1:end-2);
    n_samp = size(mydata,1);

    feature = cell(1,n_samp);
    parfor i = 1:n_samp
        data = mydata(i,:);
        spectrum = fft(data);
        spectrum = spectrum(1:N_spect(dico));
        spectrum_data = [real(spectrum)', imag(spectrum)'];
        %spectrum_norm = N_spect(dico) * spectrum_data ./ repmat(sum(abs(spectrum_data).^2),[N_spect(dico),1]);
        spectrum_norm = N_spect(dico) * spectrum_data ./ repmat(sqrt(sum(abs(spectrum_data(:)).^2)),[N_spect(dico),2]);
        data_to_model = [f', spectrum_norm];

        BIC = zeros(1,length(M_set));
        GMModels = cell(1,length(M_set));
        options = statset('MaxIter',500);
        for k = M_set
            try
                GMModels{k} = fitgmdist(data_to_model,k,'Options',options);
                BIC(k)= GMModels{k}.BIC;
            catch
                GMModels{k} = [];
                BIC(k) = inf;
            end
        end

        [minBIC,numComponents] = min(BIC);

        feature{i} = GMModels{numComponents};
        %i
    end
    disp('Finish')
    %
    save(['Features/Clean database features/Complex/T',num2str(T(dico)),'/feature_',class{n},'.mat'],'feature')
end