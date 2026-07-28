function database = create_dataset(folder, param)

fs = param.fs;
T = param.T;
D = param.D;

my_names = dir(folder);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end
n_class = length(class);

database = zeros(81202,T+2);
window = hanning(T);
variance = 0.01;
cnt = 1;
for k = 1:n_class
    disp(['-- class: ',class{k}])
    class_folder = [folder,class{k}];
    names = dir(class_folder);
    names = names(3:end);
    n_sounds = length(names);

    for i = 1:n_sounds
        file = [class_folder,'/',names(i).name];
        [sound,fs_sound] = audioread(file);
        % Test if the sound has the same sampling rate as the target
        if fs_sound ~= fs
            sound = resample(sound, fs, fs_sound);
        end
        % Mean a stereo signal to form a mono signal
        if size(sound,2) == 2
            sound = mean(sound,2);
        end
        sound = sound - mean(sound);
        % Add noise if the signal is shorter than T, and add noise before the
        % beginning and after the end
        N_aux = length(sound);
        if N_aux <= T + D
            sound = [sound; variance*max(abs(sound))*randn(D+T-N_aux,1)];
        end
        N_aux = length(sound);
        sound = sound + variance*max(abs(sound))*randn(N_aux,1);
        sound = [variance*max(abs(sound))*randn(D,1); sound; variance*max(abs(sound))*randn(D,1)];
%         p = audioplayer(sound,fs);
%         play(p)
        
        N = floor((N_aux - T) / D);
        for b = 1:N
            %database(cnt,:) = [sound((b-1)*D+1:(b-1)*D+T)',k,i];
            database(cnt,:) = [(window .* sound((b-1)*D+1:(b-1)*D+T))',k,i];
            cnt = cnt + 1;
        end
    end
end