clear

addpath(genpath(pwd))
% path = '..\Data\MUS2016\DSD100\Sources\Dev\';
path = '\\AUDIO06\ServeurMatlab\Ph. D Thesis\Data\Music\';

instruments = {'bass','drums','instrumental','guitar','vocals'};
types = {'acoustic','electro','metal'};

% bands = dir(path);
% bands = bands(3:end);
%
% n_bands = length(bands);
%
wlen = 1024;
window = hanning(wlen);
hopsize = wlen / 2;
n_fft = wlen;
F = n_fft / 2 + 1;

%% plcs on type of instrument
c_inst = 2;
c_type = [1,2];

Z = 50;
iter_max = 100;
fixed = {[],[],[]};

I = 2;


for cc = 1:2
    V = [];
    names = dir([path,instruments{c_inst},' (',types{c_type(cc)},')\']);
    names = names(3:end);
    for nn = 1:length(names)
        clc
        disp(['n: ',num2str(nn),' / ',num2str(length(names))])
        [sound,fs] = audioread([path,instruments{c_inst},' (',types{c_type(cc)},')\',names(nn).name]);
        msound = mean(sound,2);
        idx = db(abs(msound).^2) > -160;
        sound = sound(idx,:);

        spectrums = [];
        for i = 1:I
            spectrogram_sound = spectrogram(sound(:,i),sqrt(window),hopsize,n_fft);

            spectrums(:,:,i) = abs(spectrogram_sound);
            spectrums(:,:,i) = spectrums(:,:,i) ./ repmat(sum(spectrums(:,:,i),1),[F,1]);
        end
        try
            [~,P_f,~] = plcs(spectrums,Z,iter_max,fixed);
        catch
            spectrogram_sound = spectrogram(mean(sound,2),sqrt(window),hopsize,n_fft);
            spectrums = abs(spectrogram_sound);
            spectrums = spectrums ./ repmat(sum(spectrums,1),[F,1]);
            [~,P_f,~] = plca(spectrums,Z,iter_max,fixed);
        end

        V = [V; P_f];
        %     V = [V, spectrums];
        %progressbar(n,n_bands)
    end
    save(['Prototypes/Separation/Data/dict_',instruments{c_inst},'_',types{c_type(cc)},'.mat'],'V')
end   
%save(['dict_',type,'_plcs.mat'],'V')

%% plca
type = 'vocals';
Z = 50;
iter_max = 100;
fixed = {[],[],[]};

I = 2;
V = [];
for n = 1:n_bands
    clc
    disp(['n: ',num2str(n),' / ',num2str(n_bands)])
    [sound,fs] = audioread([path,bands(n).name,'\',type,'.wav']);
    %sound = mean(sound,2);
    %sound = sound(db(abs(sound).^2) > -160);
    spectrums = [];
    for i = 1:I
        spectrogram_sound = spectrogram(sound(:,i),sqrt(window),hopsize,n_fft);
        
        spectrums(:,:,i) = abs(spectrogram_sound);
        spectrums(:,:,i) = spectrums(:,:,i) ./ repmat(sum(spectrums(:,:,i),1),[F,1]);
    end
    
    [~,P_f,~] = plcs(spectrums,Z,iter_max,fixed);
    
    V = [V; P_f];
    %     V = [V, spectrums];
    %progressbar(n,n_bands)
end

save(['dict_',type,'_plcs.mat'],'V')
%%
type = 'other';
method = 'ward';
load(['dict_',type,'_plca.mat'],'V')

Y = pdist(sqrt(V/2));
Z = linkage(Y,method);

idx_clusters = cluster(Z, 'maxclust', 50);
aux = zeros(F,50);
for k = 1:50
    aux(:,k) = mean(V(idx_clusters == k,:),1);
end
rV = aux;
plot(rV)
save(['dict_',type,'_plca_r.mat'],'rV')

%% hierarchical
type = 'vocals';
method = 'ward';
n_clusters = 500;

%V = zeros(F,n_clusters);
V = [];
for n = 1:n_bands
    [sound,fs] = audioread([path,bands(n).name,'\',type,'.wav']);
    sound = mean(sound,2);
    sound = sound(db(abs(sound).^2) > -160);
    spectrogram_sound = spectrogram(sound,sqrt(window),hopsize,n_fft);
    spectrums = abs(spectrogram_sound);
    spectrums = spectrums ./ repmat(sum(spectrums,1),[F,1]);
    
    %     Y = pdist(sqrt(spectrums'/2));
    %     Z = linkage(Y,method);
    %     save(['Z_',type,'_',num2str(n),'.mat'],'Z')
    
    load(['Z_',type,'_',num2str(n),'.mat'],'Z')
    idx_clusters = cluster(Z, 'maxclust', n_clusters);
    aux = zeros(F,n_clusters);
    for k = 1:n_clusters
        aux(:,k) = mean(spectrums(:,idx_clusters == k),2);
    end
    r_spectrums = aux;
    V = [V, r_spectrums];
    %     V = [V, spectrums];
    progressbar(n,n_bands)
end

Y = pdist(sqrt(V'/2));
Z = linkage(Y,method);

idx_clusters = cluster(Z, 'maxclust', 50);
aux = zeros(F,n_clusters);
for k = 1:n_clusters
    aux(:,k) = mean(V(:,idx_clusters == k),2);
end
rV = aux;
plot(rV)
%%
save(['dict_',type,'_hierarchical2.mat'],'rV')
%%
rV = V ./ n_clusters;
save(['dict_',type,'_ps.mat'],'rV')
%%
N = size(V,2);
%V = V(:,randperm(N,N));
n_dict = 50;
n_clusters = floor(N / n_dict);

rV = [];
for nn = 1:n_dict
    rV(:,nn) = mean(V(:,(nn-1)*n_clusters+1:nn*n_clusters),2);
end

save(['dict_',type,'_g.mat'],'rV')
%%
n_spectrums = 40000;
n_clusters = 10;
n_blocs = floor(N / n_spectrums);
method = 'ward';
rV = zeros(F,n_clusters * n_blocs);

for n = 1:n_blocs
    data = V(:,(n-1)*n_spectrums+1:n*n_spectrums)';
    Y = pdist(sqrt(data/2));
    Z = linkage(Y,method);
    save(['Z_',num2str(n),'.mat'],'Z')
    idx_clusters = cluster(Z, 'maxclust', n_clusters);
    aux = [];
    for k = 1:n_clusters
        aux(k,:) = mean(data(idx_clusters == k,:),1);
    end
    rV(:,(n-1)*n_clusters+1:n*n_clusters) = aux';
    progressbar(n,n_blocs)
end