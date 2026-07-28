clear
class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};
n_class = length(class);

addpath('Statistics')

%% Construct learning set a-volute
fid = fopen('Polyphonic sounds/labels.txt');
sound_list = cell(1);
start_time = [];
end_time = [];
label = cell(1);

tline = fgetl(fid);
cnt = 1;
while ischar(tline)
    content = strsplit(tline,'\t');
    sound_list{cnt} = content{1};
    start_time(cnt) = str2double(content{2});
    end_time(cnt) = str2double(content{3});
    label{cnt} = content{4};
    cnt = cnt + 1;
    tline = fgetl(fid);
end
fclose(fid);

sound_name = unique(sound_list);
n_sound = length(sound_name);

V_train = [];
for n = 1:n_sound
    % read sound file
    [audio,sr] = audioread(['Polyphonic sounds/',sound_name{n}]);
    % file where the name is in lists
    cnt = 1;
    idx_name = [];
    for nn = 1:length(sound_list)
        name = sound_list{nn};
        if strcmp(name, sound_name{n})
            idx_name = [idx_name,cnt];
        end
        cnt = cnt + 1;
    end
    % convert to mono
    if size(audio,2) >= 2
        audio = mean(audio,2);
    end
    N_window = 1024;
    Window = hanning(N_window);
    Nshift = N_window / 2;
    Nfft = N_window;
    spectrogram_complex = spectrogram(audio,sqrt(Window),Nshift,Nfft);

    V_1 = abs(spectrogram_complex);
    [F,N] = size(V_1);
    % compute activation matrix based on onset and offset
    activation_matrix = zeros(n_class,N);
    start_time_idx = round(start_time(idx_name) * sr / Nshift);
    start_time_idx(start_time_idx == 0) = 1;
    end_time_idx = round(end_time(idx_name) * sr / Nshift);
    end_time_idx(end_time_idx > N) = N;
    label_name = [];
    for nn = 1:length(idx_name);
        label_name{nn} = label{idx_name(nn)};
    end
    for nn = 1:length(label_name)
        idx_class = find(ismember(class, label_name{nn}));
        activation_matrix(idx_class,start_time_idx(nn):end_time_idx(nn)) = 1;
    end
    V_2 = activation_matrix;
    aux_V = [V_1;V_2];
    %
    V_train = [V_train, aux_V];
end
N = size(V_train,2);
percent_train = 0.8;

idx_train = sort(randperm(N,round(N*percent_train)));
idx_test = setdiff(1:N,idx_train);

V_test = V_train(:,idx_test);
V_train = V_train(:,idx_train);

%% Construct learning set tut
data_folder = 'F:\\Ph. D Thesis\\DCASE2017\\Task 3\\TUT-sound-events-2017-development\\';
evaluation_folder = 'evaluation_setup\\';
audio_folder = 'audio\\';

classes = {'brakes squeaking', 'car', 'children', 'large vehicle', 'people speaking', 'people walking'};
n_class = length(classes);

fold = 1;

fid = fopen([data_folder,evaluation_folder,'street_fold',num2str(fold),'_train.txt']);
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

sound_name = unique(sound_list);
n_sound = length(sound_name);

V_train = [];
for n = 1:n_sound
    % read sound file
    [audio,sr] = audioread([data_folder,sound_name{n}]);
    % file where the name is in lists
    cnt = 1;
    idx_name = [];
    for nn = 1:length(sound_list)
        name = sound_list{nn};
        if strcmp(name, sound_name{n})
            idx_name = [idx_name,cnt];
        end
        cnt = cnt + 1;
    end
    % convert to mono
    if size(audio,2) >= 2
        audio = mean(audio,2);
    end
    N_window = round(sr*0.1);
    Window = hanning(N_window);
    Nshift = N_window / 2;
    Nfft = 1024;
    spectrogram_complex = spectrogram(audio,sqrt(Window),Nshift,Nfft);

    V_1 = abs(spectrogram_complex);
    [F,N] = size(V_1);
    % compute activation matrix based on onset and offset
    activation_matrix = zeros(n_class,N);
    start_time_idx = round(start_time(idx_name) * sr / Nshift);
    start_time_idx(start_time_idx == 0) = 1;
    end_time_idx = round(end_time(idx_name) * sr / Nshift);
    end_time_idx(end_time_idx > N) = N;
    label_name = [];
    for nn = 1:length(idx_name);
        label_name{nn} = label{idx_name(nn)};
    end
    for nn = 1:length(label_name)
        idx_class = find(ismember(class, label_name{nn}));
        activation_matrix(idx_class,start_time_idx(nn):end_time_idx(nn)) = 1;
    end
    V_2 = activation_matrix;
    aux_V = [V_1;V_2];
    %
    V_train = [V_train, aux_V];
    progressbar(n,n_sound)
end

%% Apply NMF with MMLE
% Initialize parameters
E = n_class;
N = size(V_train,2);
K = 10;
W = abs(randn(F+E,K)) + ones(F+E,K);
H = abs(randn(K,N)) + ones(K,N);

alpha = ones(K,1);
beta = ones(K,1);

alpha_b = ones(K,N);
beta_b = ones(K,1);
p = zeros(K,F+E,N);

iter_max = 100;
for iter = 1:iter_max
    W_old = W;
    H_old = H;
    % E-step
    log_H = psi(alpha_b) + repmat(log(beta_b),[1,N]);
    for k = 1:K
        p(k,:,:) = repmat(W_old(:,k),[1,N]) .* repmat(exp(log_H(k,:)),[F+E,1]);
    end
    p = p ./ repmat(sum(p,1),[K,1,1]);
    
    C = p .* repmat(reshape(V_train,[1,F+E,N]),[K,1,1]);
    alpha_b = repmat(alpha,[1,N]) + reshape(sum(C,2),[K,N]);
    beta_b = 1./ (1./beta + sum(W_old,1)');
    
    H = alpha_b .* repmat(beta_b,[1,N]);
    
    % M-step
    W = sum(C,3)' ./ repmat(sum(H,2)',[F+E,1]);
    
    progressbar(iter,iter_max)
end


%% Test
% Initialize parameters
N = size(V_test,2);
K = 10;
H = abs(randn(K,N)) + ones(K,N);

alpha = ones(K,1);
beta = ones(K,1);

alpha_b = ones(K,N);
beta_b = ones(K,1);
p = zeros(K,F,N);

iter_max = 100;
W_old = W(1:F,:);
for iter = 1:iter_max
    H_old = H;
    % E-step
    log_H = psi(alpha_b) + repmat(log(beta_b),[1,N]);
    for k = 1:K
        p(k,:,:) = repmat(W_old(:,k),[1,N]) .* repmat(exp(log_H(k,:)),[F,1]);
    end
    p = p ./ repmat(sum(p,1),[K,1,1]);
    
    C = p .* repmat(reshape(V_test(1:F,:),[1,F,N]),[K,1,1]);
    alpha_b = repmat(alpha,[1,N]) + reshape(sum(C,2),[K,N]);
    beta_b = 1./ (1./beta + sum(W_old,1)');
    
    H = alpha_b .* repmat(beta_b,[1,N]);
    
    progressbar(iter,iter_max)
end
%%
annotation_hat = W(F+1:F+E,:) * H;
threshold = 0.3;
annotation_hat(annotation_hat >= threshold) = 1;
annotation_hat(annotation_hat < threshold) = 0;

figure(1)
clf
imagesc(V_test(F+1:F+E,:))

figure(2)
clf
imagesc(annotation_hat)

[f1_score, error_rate] = metrics_sed(V_test(F+1:F+E,:), annotation_hat);
%%
threshold = 0:0.01:1;
f1_score = zeros(1,length(threshold));
error_rate = zeros(1,length(threshold));
for n = 1:length(threshold)
annotation_hat = W(F+1:F+E,:) * H;
annotation_hat(annotation_hat >= threshold(n)) = 1;
annotation_hat(annotation_hat < threshold(n)) = 0;

[f1_score(n), error_rate(n)] = metrics_sed(V_test(F+1:F+E,:), annotation_hat);
end

figure(3)
clf
plot(threshold,f1_score)
hold on
plot(threshold,error_rate)
legend('F1','ER')

[best_f1,idx_f1] = max(f1_score);
[best_er,idx_er] = min(error_rate);

