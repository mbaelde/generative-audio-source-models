clear
clc

data_folder = '../../Data/';
folder_database = 'ESC-10/';

startup
addpath(genpath('Database'))
addpath(genpath('Statistics'))
addpath(genpath('Tree functions'))

%% Initialisation
dico = 3;
fold = 1;
class = {'Engine', 'Detonation', 'Voice', 'Alarm', 'Step'};
n_class = length(class);
%% Load data
% load(['Database/',folder_database,'FS',num2str(fs),'/T',num2str(T(dico)),'/Uniform/80-20/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'])
% clear database_training feature_test
load(['Database/',folder_database,'FS',num2str(fs(dico)),'/T',num2str(T(dico)),'/Metaclasse/dataset_T',num2str(T(dico)),'_fold_',num2str(fold),'.mat'],'database_test','feature_training')

%
%n_class = 3;
for k = 2:n_class
    feature_class = feature_training(feature_training(:,end-1) == k,:);
    %feature_class = abs(raw_spectrum_training(raw_spectrum_training(:,end-1) == k,[1:T(dico)/2+1,T(dico)+(1:2)]));
    feature_class(:,1:end-2) = feature_class(:,1:end-2) ./ repmat(sum(feature_class(:,1:end-2),2),[1,T(dico)/2+1]);
    % Clustering
    Y = pdist(sqrt(feature_class(:,1:end-2)/2));
    method = 'ward';
    Z = linkage(Y,method);
    save(['Clusters/',folder_database,'Uniform/Fold ',num2str(fold),'/Z_',num2str(k),'.mat'],'Z')
    progressbar(k,n_class)
end
