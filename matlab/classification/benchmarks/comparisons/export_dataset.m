clear
clc
folder_data = 'F:\Ph. D Thesis\DCASE2017\Task 3\TUT-sound-events-2017-development\';
folder_audio = 'audio\';
folder_evaluation = 'evaluation_setup\';

distcomp.feature( 'LocalUseMpiexec', false )

%% Define parameters for mixture models estimation of the spectrum
fs = 44100;                                         % Sampling rate
T = 2048;
D = 2048;                                            % Time interval between two analysys windows (sample)
N_fft = T;                                          % FFT size of the analysis window
N_spect = round(N_fft/5);                           % Number of points kept in the spectrum

%% Construct train data
fold = 1;

fid = fopen([folder_data,folder_evaluation,'fold',num2str(fold),'_train.txt']);
tline = fgetl(fid);
cnt = 1;
while ischar(tline)
    mstr{cnt} = strsplit(tline,'\t');
    name{cnt} = mstr{cnt}{1};
    classes{cnt} = mstr{cnt}{2};
    cnt = cnt + 1;
    tline = fgetl(fid);
end
fclose(fid);

n_class = length(unique(classes));

u_classes = unique(classes);

n_sound = length(name);
dict_size = 214*n_sound;
feature_training = zeros(dict_size,N_spect+1);
offset = 0;
for n = 1:n_sound
    aux = strfind(u_classes,classes{n});
    for k = 1:n_class
        if ~isempty(aux{k})
            idx_class = k;
        end
    end
    [sound,fs] = audioread([folder_data,name{n}]);
    sound = mean(sound');
    N = length(sound); 
    n_buff = floor((N-T)/D);
    for b = 1:n_buff
        database = [sound((b-1)*D+1:(b-1)*D+T),idx_class];
        spectrum = fft(database(1:end-1));
        spectrum = abs(spectrum).^2;
        spectrum = spectrum(1:N_spect) ./ sum(spectrum(1:N_spect));
        feature_training(b+offset,:) = [spectrum, database(end)];
    end
    offset = offset + n_buff;
    clc
    disp(['n: ',num2str(n),' / ',num2str(n_sound)])
end

save('feature_training.mat','feature_training','-v7.3')

%% Construct test data
fold = 1;

fid = fopen([folder_data,folder_evaluation,'fold',num2str(fold),'_evaluate.txt']);
tline = fgetl(fid);
cnt = 1;
while ischar(tline)
    mstr{cnt} = strsplit(tline,'\t');
    name{cnt} = mstr{cnt}{1};
    classes{cnt} = mstr{cnt}{2};
    cnt = cnt + 1;
    tline = fgetl(fid);
end
fclose(fid);

n_class = length(unique(classes));

u_classes = unique(classes);

n_sound = length(name);
dict_size = 857*n_sound;
database_test = zeros(dict_size,T+1);
offset = 0;
for n = 1:n_sound
    aux = strfind(u_classes,classes{n});
    for k = 1:n_class
        if ~isempty(aux{k})
            idx_class = k;
        end
    end
    [sound,fs] = audioread([folder_data,name{n}]);
    sound = mean(sound');
    N = length(sound);
    n_buff = floor((N-T)/D);
    for b = 1:n_buff
        database_test(b+offset,:) = [sound((b-1)*D+1:(b-1)*D+T),idx_class];
    end
    offset = offset + n_buff;
    clc
    disp(['n: ',num2str(n),' / ',num2str(n_sound)])
end

save('database_test.mat','database_test','-v7.3')

