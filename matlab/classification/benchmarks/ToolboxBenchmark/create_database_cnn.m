fs = 44100;                    % Sampling rate

class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
n_class = length(class);

data_folder = '../../Data/';

n_max = 0.5 * fs;

for k = 3:n_class
    cnt = 1;
    database = [];
    aux_sound = [];
    disp(['Currently: class ', class{k}])
    folder = [data_folder,class{k}];
    names = dir(folder);
    names = names(3:end);
    for i = 1:length(names)
        file = [folder,'/',names(i).name];
        [sound,fs_sound] = audioread(file);
        % Test if the sound has the same sampling rate as the target
        f_factor = fs / fs_sound;
        if f_factor ~= 1
            sound = resample(sound, f_factor, 1);
        end
        % Mean a stereo signal to form a mono signal
        if size(sound,2) == 2
            sound = mean(sound,2);
        end
        sound = sound - mean(sound);
        sound = [0.05*max(abs(sound))*randn(512,1); sound; 0.05*max(abs(sound))*randn(512,1)];
        
        if size(sound,1) > size(sound,2)
            sound = sound';
        end
        
        sound = [aux_sound, sound];
        
        N_test = length(sound);
        
        if N_test < n_max
            aux_sound = [aux_sound, sound];
        else
            n_seg = floor(N_test / n_max);
            for seg = 1:n_seg
                database(cnt,:) = [sound(1+(seg-1)*n_max:(seg)*n_max),k,i];
                aux_sound = [];
                cnt = cnt + 1;
            end
        end
        
        progressbar(i,length(names))
    end
    
    save(['Database/CNN/database_',class{k},'.mat'],'database')
end

aux_database = [];
for k = 1:n_class
    load(['Database/CNN/database_',class{k},'.mat'])
    aux_database = [aux_database; database];
end
database = aux_database;
save(['Database/CNN/database.mat'],'database', '-v7.3')
%end