clear
clc

data_folder = '../../Data/';
database_folder = 'A-Volute/';

startup
addpath(genpath('Database'))
addpath(genpath('Statistics'))
addpath(genpath('Tree functions'))

%% Initialisation
dico = 3;
fold = 1;

%% Load data
load(['Database/',database_folder,'FS',num2str(fs),'/T',num2str(T(dico)),'/Uniform/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'])
clear database_training feature_training

database = database_test;
aux_L = feature_test;

T = size(database,2)-2;
N_spect = size(aux_L,2)-2;

aux_L(:,1:end-2) = aux_L(:,1:end-2) ./ repmat(sum(aux_L(:,1:end-2),2),[1,N_spect]);

class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
n_class = length(class);

for k = 1:n_class
    idx{k} = aux_L(:,end-1) == k;
    aux_L_class{k} = aux_L(idx{k},:);
    database_class{k} = database(idx{k},:);
end
clear aux_L database
%% Choose data
g = [6,8]; % indices of the class

c = {[1:50],[1,2,3,4]}; % indices of the sounds per class
% Find the indices of the models for each sound in c
idx_sound = cell(1,length(g));
n_models = cell(1,length(g));
dict_size = zeros(1,length(g));
for gg = 1:length(g)
    for cc = 1:length(c{gg})
        idx_sound{gg}{cc} = find(database_class{g(gg)}(:,end) == c{gg}(cc));
        n_models{gg} = [n_models{gg},length(idx_sound{gg}{cc})]; % how many models per sounds
    end
    dict_size(gg) = sum(n_models{gg});
end

% Indices to get the models
theta = cell(1,length(g));
for gg = 1:length(g)
    for mm = 1:length(n_models{gg})
        theta{gg}{mm} = 1:n_models{gg}(mm);
    end
end
dict_size
n_comb = prod(dict_size)
%%
c_models = [0,cumsum(dict_size)];
% Allocate the memory for each models and every combination
test_sound = zeros(sum(dict_size)+n_comb,T);
aux_L = zeros(sum(dict_size)+n_comb,N_spect);
% Fill the individual models
for gg = 1:length(g)
    aux_test = [];
    aux_aux = [];
    for cc = 1:length(c{gg})
%         test_sound(c_models(cc)+1:c_models(cc+1),:) = database_class{g(gg)}(idx_sound{gg}{cc}(theta{gg}{cc}),1:T);
%         aux_L(c_models(cc)+1:c_models(cc+1),:) = aux_L_class{g(cc)}(idx_sound{cc}(theta{cc}),1:N_spect);
        aux_test = [aux_test; database_class{g(gg)}(idx_sound{gg}{cc}(theta{gg}{cc}),1:T)];
        aux_aux = [aux_aux; aux_L_class{g(gg)}(idx_sound{gg}{cc}(theta{gg}{cc}),1:N_spect)];
    end
    aux_test = aux_test ./ repmat(max(abs(aux_test),[],2),[1,T]);
    test_sound(c_models(gg)+1:c_models(gg+1),:) = aux_test;
    aux_L(c_models(gg)+1:c_models(gg+1),:) = aux_aux;
end
% Fill every combination
offset = sum(dict_size);
cnt = offset;
for i = 1:dict_size(1)
    test_sound(cnt+1:cnt+dict_size(2),:) = 0.5 * (repmat( test_sound(i,:), [dict_size(2),1] ) + test_sound(c_models(2)+1:c_models(3),:));
    aux_L(cnt+1:cnt+dict_size(2),:) = 0.5 * (repmat( aux_L(i,:), [dict_size(2),1] ) + aux_L(c_models(2)+1:c_models(3),:));
    cnt = cnt + dict_size(2);
end
% for gg = 1:length(g)
%     aux_test = [];
%     aux_aux = [];
%     for cc = 1:length(c{gg})
% %         test_sound(offset+(i-1)*n_models(2)+1:offset+i*n_models(2),:) = 0.5 * repmat(test_sound(i,:),[n_models(2),1]) + 0.5 * test_sound(c_models(2)+1:c_models(3),:);
% %         aux_L(offset+(i-1)*n_models(2)+1:offset+i*n_models(2),:) = 0.5 * repmat(aux_L(i,:),[n_models(2),1]) + 0.5 * aux_L(c_models(2)+1:c_models(3),:);
%     end
% end

dict_size = [dict_size,n_comb];
aux_L_comp = gpuArray(log(aux_L));
%%
n_buff = 1;

cnt = 0;
msize = [0,cumsum(dict_size)];
n_sounds = [length(c{1}),length(c{2})];
n_sounds = [n_sounds, sum(n_sounds)];
prior_g = log(dict_size ./ sum(dict_size));
%prior_g = log([1/3,1/3,1/3]);

% Compute probabilities
L = gpuArray(zeros(n_buff,sum(dict_size)));
L_prior_max = gpuArray(zeros(1,length(msize)-1));
L_bay = zeros(1,size(test_sound,1));
likelihood_group = gpuArray(zeros(1,length(g)+1));
posterior_g = gpuArray(zeros(n_buff,3));
tic
for ii = 1:size(test_sound,1)
    cnt = cnt + 1;
    spectrum = abs(fft(test_sound(ii,:))).^2;
    spectrum = N_spect * spectrum(1:N_spect) ./ sum(spectrum(1:N_spect),2);
    spectrum_norm = gpuArray(repmat(spectrum,[msize(end),1]));
    % Model likelihood
    L(cnt,:) = sum(spectrum_norm .* aux_L_comp,2);
    % Group likelihood    
    for gg = 1:length(msize)-1
        A = L(cnt,msize(gg)+1:msize(gg+1));
        L_prior_max(gg) = max(A);
        likelihood_group(gg) = log(sum(exp(A - L_prior_max(gg))));
    end
    likelihood_group = likelihood_group + L_prior_max - log(n_sounds);
    
    A = likelihood_group + prior_g;
    L_post_max = max(A);
    norm_factor_g = L_post_max + log(sum(exp(A - L_post_max)));
    posterior_g(cnt,:) = -norm_factor_g + likelihood_group + prior_g;
    
    if mod(ii,n_buff) == 0
        cnt = 0;
        sum_g = sum(gather(posterior_g),1);
        L_bay((ii-n_buff+1):ii) = find(max(sum_g) == sum_g);
    end
    %progressbar(ii,size(test_sound,1));
    clc
    disp(['iter : ',num2str(ii),' / ',num2str(size(test_sound,1))])
end
elapsed_time = toc;

idx_p = L_bay > 0;
L_bay = L_bay(idx_p);
true_class = [1*ones(1,dict_size(1)),2*ones(1,dict_size(2)),3*ones(1,dict_size(3))];
true_class = true_class(idx_p);
        
%save('tmp_mixture.mat','L_bay');
confusion_matrix = confusionmat(true_class, L_bay, 'order', 1:3);
BAY_confusionmatrix = confusion_matrix ./ repmat(sum(confusion_matrix,2),[1,3])

%
save(['Results/Mixture/Exhaustive/result_',num2str(g(1)),'_',num2str(g(2)),'.mat'],'g','c','L_bay','BAY_confusionmatrix')
%% Try to recover an unknown mix of sounds with EM algorithm or Newton's method
M = size(database_test,1);
prop_true = rand(M,1);
prop_true = prop_true ./ sum(prop_true);

mixture_sound = zeros(1,T);

for i = 1:M
    mixture_sound = mixture_sound + prop_true(i) * (database_test(i,1:end-2) ./ max(abs(database_test(i,1:end-2))));
end

prop_em = ones(M,1) / M;

stop = 0;
iter_max = 10;
iter_cur = 0;
L(1) = inf;


%%
N = 500;
aux_L_mixed = zeros(N*size(aux_L_voice_female,1),205);

for n = 1:N
    aux_L_mixed(1+(n-1)*size(aux_L_voice_female,1):n*size(aux_L_voice_female,1),:) = 0.5*repmat(aux_L_gunshot(n,1:205),[size(aux_L_voice_female,1),1]) + 0.5*aux_L_voice_female(:,1:205);
    progressbar(n,N)
end


%% Construct models
clc
fs = 44100;
N_fft = 1024;
N_spect = round(N_fft / 5);
f = 0:fs/N_fft:fs/2;

sound_1 = database(820,1:1024);
sound_1 = sound_1 / max(sound_1);
sound_2 = database(74916,1:1024);
sound_2 = sound_2 / max(sound_2);

mixture_sound = 0.5 * sound_1 + 0.5 * sound_2;

figure(1)
clf
plot(0.5*sound_1)
hold on
plot(0.5*sound_2)
plot(mixture_sound)
legend('Sound 1', 'Sound 2', 'Mixture')

spectrum_1 = abs(fft(sound_1)).^2;
spectrum_1 = N_spect * spectrum_1(1:N_spect) / sum(spectrum_1(1:N_spect));
spectrum_2 = abs(fft(sound_2)).^2;
spectrum_2 = N_spect * spectrum_2(1:N_spect) / sum(spectrum_2(1:N_spect));

mixture_spectrum = abs(fft(mixture_sound)).^2;
mixture_spectrum = N_spect * mixture_spectrum(1:N_spect) / sum(mixture_spectrum(1:N_spect));

figure(2)
clf
plot(0.5*spectrum_1)
hold on
plot(0.5*spectrum_2)
plot(mixture_spectrum)
legend('Sound 1', 'Sound 2', 'Mixture')

% Calcul des vraisemblances
model_1 = feature_airplane{2}{1}.spectrum_model;
model_2 = feature_voice_female{2}{4}.spectrum_model;

cdf_1 = model2cdf(model_1, f(1:N_spect+1),0);
cdf_2 = model2cdf(model_2, f(1:N_spect+1),0);
cdf_mixture = 0.5 * cdf_1 + 0.5 * cdf_2;

L_simple = sum(repmat(mixture_spectrum,[size(aux_L,1),1]) .* aux_L(:,1:end-2),2);
L_2 = sum(mixture_spectrum .* log(diff(cdf_2)));
L_mixture = sum(mixture_spectrum .* log(diff(cdf_mixture)));
%% Construct a mixed pdf
prop = [0.7,0.3];

mixed_pdf = prop(1) * pdf_1 + prop(2) * pdf_2;

prop_mat = [0.1,0.9;
            0.2,0.8;
            0.3,0.7;
            0.4,0.6;
            0.5,0.5;
            0.6,0.4;
            0.7,0.3;
            0.8,0.2;
            0.9,0.1];
        
cdf_1 = diff(model2cdf(model_1, f(1:N_spect+1),0));
cdf_2 = diff(model2cdf(model_2, f(1:N_spect+1),0));
    
for i = 1:9
    % EM algorithm to recover the proportion
    prop_em(i,:) = prop_mat(i,:);

    stop = 0;

    L(i) = inf;
    iter = 0;
    max_iter = 100;
    tol = 1e-6;
    while ~stop
        L_old(i) = L(i);

        % E step   
        mixture_cdf = (prop_em(i,1) * cdf_1) + (prop_em(i,2) * cdf_2);

        E_tau = [prop_em(i,1) * cdf_1; 
                 prop_em(i,2) * cdf_2] ./ repmat( mixture_cdf , [2,1]) ;

        % M step
        prop_em(i,:) = sum(repmat(mixed_pdf,[2,1]) .* E_tau, 2) ./ sum(repmat(mixed_pdf,[2,1]),2);

        % Stopping criterion
        L(i) = sum( mixed_pdf .* log( prop_em(i,1) * cdf_1 + prop_em(i,2) * cdf_2 ) );
        iter = iter + 1;

        stop = ((abs(L(i) - L_old(i)) / abs(L(i))) < tol) || iter > max_iter;
    end
    BIC(i) = -2*L(i) + 2*log(N_spect);
end

[~,i_max] = max(BIC);
disp(['Best match: ',num2str(i_max),' ; prop: ',num2str(prop_em(i_max,1)),', ',num2str(prop_em(i_max,2))])

resynthesis = prop_em(i_max,1) * pdf_1 + prop_em(i_max,2) * pdf_2;

figure(10)
clf
plot(f(1:N_spect), prop(1)*pdf_1)
hold on
plot(f(1:N_spect), prop(2)*pdf_2)
plot(f(1:N_spect), mixed_pdf)
plot(f(1:N_spect), resynthesis)
legend('Spectre 1', 'Spectre 2', 'Mixed spectre', 'Spectre em')
%% Mixture of mixtures of gaussians
moy = [0,6 ; 3,8];
sig = [0.5,1 ; 1,1];
p = [0.5,0.5; 0.5, 0.5 ; 0.9, 0.1];

N = 5000;

x_gauss_1 = mixture_randn(N, moy(1,:), sig(1,:), p(1,:));
x_gauss_2 = mixture_randn(N, moy(2,:), sig(2,:), p(2,:));

x_mixed_gauss = mixture_mixture_randn(N, moy, sig, p(1:2,:), p(3,:));

x = min(x_mixed_gauss):0.01:max(x_mixed_gauss);

gauss_1 = mixture_normpdf(x, moy(1,:), sig(1,:), p(1,:));
gauss_2 = mixture_normpdf(x, moy(2,:), sig(2,:), p(2,:));

mixed_gauss = mixture_mixture_normpdf(x, moy, sig, p(1:2,:), p(3,:));

[h,c] = hist(x_mixed_gauss,51);
delta = c(2)-c(1);
h = h ./ sum(h) / delta;

nmixed_gauss = mixed_gauss / max(mixed_gauss);

figure(1)
clf
bar(c,h)
hold on
plot(x,nmixed_gauss)
xlabel('x')
ylabel('probability')

%
prop_em = [0.5,0.5];

cgauss_1 = mixture_normpdf(x_mixed_gauss, moy(1,:), sig(1,:), p(1,:));
cgauss_2 = mixture_normpdf(x_mixed_gauss, moy(2,:), sig(2,:), p(2,:));

stop = 0;

L = inf;
iter = 0;
max_iter = 100;
tol = 1e-6;
while ~stop
    L_old = L;
    % E step
    norm_factor = prop_em(1) * cgauss_1 + prop_em(2) * cgauss_2;
    z_ik = [prop_em(1) * cgauss_1 ; prop_em(2) * cgauss_2] ./ repmat(norm_factor,[2,1]);
    
    % M step
    prop_em = sum(z_ik,2) / N;
    
    % Stopping criterion
    L = sum(log(prop_em(1) * gauss_1 + prop_em(2) * gauss_2));
    iter = iter + 1;
    
    stop = ((abs(L - L_old) / abs(L)) < tol) || iter > max_iter;
end

resynthesis = prop_em(1) * gauss_1 + prop_em(2) * gauss_2;

figure(10)
clf
plot(x,prop_em(1)*gauss_1)
hold on
plot(x,prop_em(2)*gauss_2)
plot(x,mixed_gauss)
plot(x,resynthesis)
xlabel('x')
ylabel('pdf')
legend('Mixture 1', 'Mixture 2', 'Mixture of Mixture', 'Resynthesis')

%% M GMM intervals
moy = [0,6 ; 3,8];
sig = [0.5,1 ; 1,1];
weight = [0.5,0.5; 0.4, 0.6]; 
prop_mixture = [0.1, 0.9];

N = 5000;

x_gauss_1 = mixture_randn(N, moy(1,:), sig(1,:), weight(1,:));
x_gauss_2 = mixture_randn(N, moy(2,:), sig(2,:), weight(2,:));

x_mixed_gauss = mixture_mixture_randn(N, moy, sig, weight, prop_mixture);

x = min(x_mixed_gauss)-10:0.01:max(x_mixed_gauss)+10;

gauss_1 = mixture_normpdf(x, moy(1,:), sig(1,:), weight(1,:));
gauss_2 = mixture_normpdf(x, moy(2,:), sig(2,:), weight(2,:));

mixed_gauss = mixture_mixture_normpdf(x, moy, sig, weight, prop_mixture);

[x_obs,c] = hist(x_mixed_gauss,51);
delta = c(2)-c(1);
h = x_obs ./ sum(x_obs) / delta;

intervals = zeros(1,length(c)+1);
intervals(1) = c(1) - (delta / 2);
intervals(2:length(c)+1) = c + (delta / 2);

nmixed_gauss = max(h) * mixed_gauss / max(mixed_gauss);

figure(1)
clf
bar(c,h)
hold on
plot(x,nmixed_gauss, 'r','LineWidth', 3)
plot(x,prop_mixture(1)*gauss_1, 'c')
plot(x,prop_mixture(2)*gauss_2, 'g')
xlabel('x')
ylabel('probability')

%
prop_em = [0.5,0.5];

model_1_gmm.mu = moy(1,:);
model_1_gmm.sigma = sig(1,:).^2;
model_1_gmm.mixing_coeff = weight(1,:);
model_2_gmm.mu = moy(2,:);
model_2_gmm.sigma = sig(2,:).^2;
model_2_gmm.mixing_coeff = weight(2,:);
%cdf_1 = mixture_normcdf(intervals, moy(1,:), sig(1,:), weight(1,:));
%cdf_2 = mixture_normcdf(intervals, moy(2,:), sig(2,:), weight(2,:));
cdf_1 = diff(model2cdf(model_1_gmm,intervals,0));
cdf_2 = diff(model2cdf(model_2_gmm,intervals,0));

stop = 0;

L = inf;
iter = 0;
max_iter = 100;
tol = 1e-6;
while ~stop
    L_old = L;
    
    % E step   
    mixture_cdf = (prop_em(1) * cdf_1) + (prop_em(2) * cdf_2 );
    
    E_tau = [prop_em(1) * cdf_1 ; 
             prop_em(2) * cdf_2 ] ./ repmat( mixture_cdf , [2,1]) ;
    
    % M step
    prop_em = sum(repmat(x_obs,[2,1]) .* E_tau, 2) ./ sum(repmat(x_obs,[2,1]),2);
    
    % Stopping criterion
    L = sum( x_obs .* log( prop_em(1) * cdf_1 + prop_em(2) * cdf_2  ) );
    iter = iter + 1;
    
    stop = ((abs(L - L_old) / abs(L)) < tol) || iter > max_iter;
end

resynthesis = prop_em(1) * gauss_1 + prop_em(2) * gauss_2;

figure(10)
clf
plot(x,prop_em(1)*gauss_1)
hold on
plot(x,prop_em(2)*gauss_2)
plot(x,mixed_gauss)
plot(x,resynthesis)
xlabel('x')
ylabel('pdf')
legend('Mixture 1', 'Mixture 2', 'Mixture of Mixture', 'Resynthesis')
