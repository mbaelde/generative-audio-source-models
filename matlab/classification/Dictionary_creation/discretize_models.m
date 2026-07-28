clear
clc
addpath(genpath('Features'))
addpath(genpath('Database'))
addpath(genpath('Statistics'))
%%
class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};

T = [512,1024,2048];
N_fft = T;
N_spect = round(N_fft/5);

dico = 2;
load(['Features/Clean database features/Complex/T',num2str(T(dico)),'/feature_T',num2str(T(dico)),'.mat'])
load(['Database/T',num2str(T(dico)),'/database_overlap.mat'])

% load('Results\Performance\Complex\err_T1024')
% cnt = 0;
% idx = setdiff(1:length(feature),err);
% cnt_feat = 1;
% for i = 1:length(feature)
%     if any(i == idx)
%         myfeature{cnt_feat} = feature{i};
%         cnt_feat = cnt_feat + 1;
%     end
% end
% feature = myfeature;
% clear myfeature

fs = 44100;
f = 0:fs/N_fft(dico):fs/2;
f = f(1:N_spect(dico));

% Search for the min and max of each dimension
for ii = 1:length(feature)
    data = database(ii,1:end-2)';
    spectrum = fft(data);
    spectrum = spectrum(1:N_spect(dico));
    spectrum_data = [real(spectrum), imag(spectrum)];
    spectrum_norm = N_spect(dico) * spectrum_data ./ repmat(sqrt(sum(abs(spectrum_data(:)).^2)),[N_spect(dico),2]);
    data_to_reco = [f', spectrum_norm];
    x_min(ii) = min(data_to_reco(:,1));
    x_max(ii) = max(data_to_reco(:,1));
    y_min(ii) = min(data_to_reco(:,2));
    y_max(ii) = max(data_to_reco(:,2));
    z_min(ii) = min(data_to_reco(:,3));
    z_max(ii) = max(data_to_reco(:,3));
end
%%
% Use 10 cells per dimension
n_x = 15;
n_y = 15;
n_z = 15;

% Use the median of each minimum and maximum for each dimension
xmin = median(x_min);
xmax = median(x_max);

ymin = median(y_min);
ymax = median(y_max);

zmin = median(z_min);
zmax = median(z_max);

% Create grid
[Y_a,X_a,Z_a] = meshgrid(linspace(ymin,ymax,n_y+1),linspace(xmin,xmax,n_x+1),linspace(zmin,zmax,n_z+1));

% Compute the number of elements in each cell
p_x = floor(size(X_a,1) / n_x);
p_y = floor(size(X_a,2) / n_y);
p_z = floor(size(X_a,3) / n_z);

% % Compute the approximate grid
% X_a = mean(reshape(X(1:end-1,1,1)',[length(X(1:end-1,1,1))/n_x,n_x]),1);
% Y_a = mean(reshape(Y(1,1:end-1,1),[length(Y(1,1:end-1,1))/n_y,n_y]),1);
% aux_Z = reshape(Z(:,1,:),size(X(:,:,1)));
% Z_a = mean(reshape(aux_Z(1,1:end-1,1),[length(Z(1,1:end-1,1))/n_z,n_z]),1);
delta_x = X_a(2,1,1)-X_a(1,1,1);
delta_y = Y_a(1,2,1)-Y_a(1,1,1);
delta_z = Z_a(1,1,2)-Z_a(1,1,1);
% aux_X = [X_a - delta_x/2, X_a(end)+delta_x/2];
% X_a = repmat([X_a - delta_x/2, X_a(end)+delta_x/2],[n_x+1,1,n_x+1]);
% Y_a = repmat([Y_a - delta_y/2, Y_a(end)+delta_y/2]',[1,n_y+1,n_y+1]);
% Z_a = permute(repmat([Z_a - delta_z/2, Z_a(end)+delta_z/2],[n_y+1,1,n_y+1]),[1,3,2]);

% Precompute the start and end points for the computation of the CDFs
cnt = 1;
for i = 1:n_x
    for j = 1:n_y
        for k = 1:n_z
            start_point = [X_a(i,j,k),Y_a(i,j,k),Z_a(i,j,k)];
            end_point = [X_a(i+1,j+1,k+1),Y_a(i+1,j+1,k+1),Z_a(i+1,j+1,k+1)];
            my_point(cnt,:) = (end_point + start_point)/2;
            cnt = cnt + 1;
        end
    end
end
%%
%gcp
l_mcdf = zeros(90000,n_x*n_y*n_z);
vol = delta_x * delta_y * delta_z;

cnt = 1;
err = [];
% Discretize the models
for ii = 1:length(feature) 
    try
        mcdf = mixture_mvnpdf(my_point,feature{ii}.mu,feature{ii}.Sigma,feature{ii}.ComponentProportion)*vol;
        l_mcdf(cnt,:) = log(mcdf);
        cnt = cnt + 1;
    catch
        err = [err,ii];
    end
    
    progressbar(ii,length(feature))
end

edges{1} = reshape(X_a(:,1,1),[n_x+1,1]);
edges{2} = reshape(Y_a(1,:,1),[n_y+1,1]);
edges{3} = reshape(Z_a(1,1,:),[n_z+1,1]);
%%
save(['Features\Clean database features\Complex\T',num2str(T(dico)),'/preprocess_dictionnary_T',num2str(T(dico)),'_',num2str(n_x),'x',num2str(n_y),'x',num2str(n_z),'.mat'],'l_mcdf','err', 'edges')

%%
load(['Database\T',num2str(T(dico)),'\database_overlap.mat'])

split_dataset_folds(l_mcdf, database, 0.8, T(dico), 'Models')
save(['Features\Clean database features\Complex\T',num2str(T(dico)),'\edges.mat'],'edges')