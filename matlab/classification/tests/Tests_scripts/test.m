clear
clc

data_folder = '../../Data/';
database_folder = 'ESC-50/';

startup
addpath(genpath('Database'))
addpath(genpath('Statistics'))
addpath(genpath('Tree functions'))
addpath(genpath('Identification procedure'))
addpath(genpath('Dictionary creation'))

dico = 3;
fold = 1;

param.T = T;
param.N_spect = N_spect;
param.fs = fs;
param.type = 'single';
param.n_class = n_class;
%%
identification_reduced_dictionary(database_folder, dico, fold, param)