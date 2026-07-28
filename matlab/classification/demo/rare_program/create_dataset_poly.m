function [database,activation_matrix, class, gm] = create_dataset_poly(file_name, audio_folder, param)
%%
T = param.T;
D = param.D;
fs = param.fs;

fid = fopen([audio_folder,file_name,'.txt']);
sound_list = cell(1);
start_time = [];
end_time = [];
label = cell(1);

tline = fgetl(fid);
cnt = 1;
while ischar(tline)
    content = strsplit(tline,'\t');
    sound_list{cnt} = content{1};
    start_time(cnt) = str2double(content{3});
    end_time(cnt) = str2double(content{4});
    label{cnt} = content{5};
    cnt = cnt + 1;
    tline = fgetl(fid);
end
fclose(fid);

class = unique(label);
n_class = length(class);

if n_class > 3
    gm = repmat((1:n_class)',[1,4]);
else
    gm = repmat((1:n_class)',[1,3]);
end

% Mixture of 2 classes
cnt = n_class+1;
for k = 1:n_class
    for m = k+1:n_class
        if n_class > 3
            gm(cnt,:) = [k,m,m,m];
        else
            gm(cnt,:) = [k,m,m];
        end
        cnt = cnt + 1;
    end
end
% Mixture of 3 classes
for k = 1:n_class
    for m = k+1:n_class
        for n = m+1:n_class
            if n_class > 3
                gm(cnt,:) = [k,m,n,n];
            else
                gm(cnt,:) = [k,m,n];
            end
            cnt = cnt + 1;
        end
    end
end
if n_class > 3
    % Mixture of 4 classes
    for k = 1:n_class
        for m = k+1:n_class
            for n = m+1:n_class
                for p = n+1:n_class
                    gm(cnt,:) = [k,m,n,p];
                    cnt = cnt + 1;
                end
            end
        end
    end
end
sound_name = unique(sound_list);
n_sound = length(sound_name);

N = 0;
for n = 1:n_sound
    % read sound file
    try
        [sound,fs_sound] = audioread([audio_folder,file_name,'.wav']);
    catch
        [sound,fs_sound] = audioread([audio_folder,'../',sound_name{n}]);
    end
    % resample if needed
    if fs_sound ~= fs
        sound = resample(sound, fs, fs_sound);
    end
    % convert to mono
    if size(sound,2) >= 2
        sound = mean(sound,2);
    end
    N_aux = length(sound);
    N = N + floor((N_aux - T) / D);
end

database = zeros(N,T+2);
activation_matrix = zeros(n_class,N);
n_mixt2 = nchoosek(n_class,2);
n_mixt3 = nchoosek(n_class,3);
if n_class > 3
n_mixt4 = nchoosek(n_class,4);
end
cnt_g = 0;
for n = 1:n_sound
    % read sound file
    try
        [sound,fs_sound] = audioread([audio_folder,file_name,'.wav']);
    catch
        [sound,fs_sound] = audioread([audio_folder,'../',sound_name{n}]);
    end
    % resample if needed
    if fs_sound ~= fs
        sound = resample(sound, fs, fs_sound);
    end
    % convert to mono
    if size(sound,2) >= 2
        sound = mean(sound,2);
    end
    N_aux = length(sound);
    N = floor((N_aux - T) / D);
    % file where the name is in lists
    cnt_n = 1;
    idx_name = [];
    for nn = 1:length(sound_list)
        name = sound_list{nn};
        if strcmp(name, sound_name{n})
            idx_name = [idx_name,cnt_n];
        end
        cnt_n = cnt_n + 1;
    end
    % compute activation matrix based on onset and offset
    tmp_activation_matrix = zeros(n_class,N);
    start_time_idx = round(start_time(idx_name) * fs_sound / D);
    start_time_idx(start_time_idx == 0) = 1;
    end_time_idx = round(end_time(idx_name) * fs_sound / D);
    end_time_idx(end_time_idx > N) = N;
    label_name = [];
    for nn = 1:length(idx_name);
        label_name{nn} = label{idx_name(nn)};
    end
    for nn = 1:length(label_name)
        idx_class = find(ismember(class, label_name{nn}));
        tmp_activation_matrix(idx_class,start_time_idx(nn):end_time_idx(nn)) = 1;
    end
    
    aux_active = zeros(size(tmp_activation_matrix));
    frames = zeros(N,T+2);
    for b = 1:N
        if any(tmp_activation_matrix(:,b) == 1)
            aux_active(:,b) = tmp_activation_matrix(:,b);
            pres_class = find(aux_active(:,b) == 1);
            if length(pres_class) == 1
                idx_class = gm(pres_class,1);
            elseif length(pres_class) == 2
                c_gm = gm(n_class+1:n_class+n_mixt2,1:2);
                idx_class = find(sum(repmat(pres_class',[n_mixt2,1]) == c_gm,2) == 2) + n_class;
            elseif length(pres_class) == 3
                c_gm = gm(n_class+n_mixt2+1:n_class+n_mixt2+n_mixt3,1:3);
                idx_class = find(sum(repmat(pres_class',[n_mixt3,1]) == c_gm,2) == 3) + n_class + n_mixt2;
            elseif length(pres_class) == 4
                c_gm = gm(n_class+n_mixt2+n_mixt3+1:n_class+n_mixt2+n_mixt3+n_mixt4,:);
                idx_class = find(sum(repmat(pres_class',[n_mixt4,1]) == c_gm,2) == 4) + n_class + n_mixt2 + n_mixt3;
            end
            frames(b,:) = [sound((b-1)*D+1:(b-1)*D+T)',idx_class,n];
        else
            frames(b,:) = [sound((b-1)*D+1:(b-1)*D+T)',0,n];
        end
    end
    activation_matrix(:,cnt_g+1:cnt_g+N) = aux_active;
    database(cnt_g+1:cnt_g+N,:) = frames;
    cnt_g = cnt_g + N;
end