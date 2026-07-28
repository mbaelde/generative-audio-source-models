function model = ninterval_spectrum_model(spectrum, N_spect, M_set, f, criterion)
%% ninterval_spectrum_model
%
% This function returns a mixture model modelizing the energy spectrum. It
% takes the spectrum data sampled at fs, and model the N_spect-points
% spectrum. M_set represents the set of candidates number of mixture 
% components, f is a vector of frequencies, verbose is a flag to display
% information and criterion is either 'AIC' or 'BIC'.
%
% model = spectrum_model(data, fs, N_spect, M_set, f, verbose, criterion)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

if size(spectrum,1) > size(spectrum,2)
    spectrum = spectrum';
end

spectrum_norm = N_spect * spectrum(1:N_spect) / sum(spectrum(1:N_spect));

count_f = spectrum2hist(spectrum_norm, f(1:N_spect));

if strcmp(criterion,'AIC')
    AIC = zeros(length(M_set),1);
elseif strcmp(criterion,'BIC')
    BIC = zeros(length(M_set),1);
end

GMModels = cell(1,length(M_set));
options = statset('MaxIter',500,'Display','off');
parfor k = M_set
    try
        GMModels{k} = fitgmdist(count_f,k,'Options',options, 'Replicates',5);
        if strcmp(criterion,'AIC')
            AIC(k)= GMModels{k}.AIC;
        elseif strcmp(criterion,'BIC')
            BIC(k)= GMModels{k}.BIC;
        end
    catch
        GMModels{k} = [];
        if strcmp(criterion,'AIC')
            AIC(k)= inf;
        elseif strcmp(criterion,'BIC')
            BIC(k)= inf;
        end
    end
end

[~,numComponents] = min(BIC);

model.n_components = M_set(numComponents);
model.mu = GMModels{numComponents}.mu;
model.sigma = reshape(GMModels{numComponents}.Sigma,[numComponents,1]);
model.mixing_coeff = GMModels{numComponents}.ComponentProportion';
model.iter = GMModels{numComponents}.NumIterations;
model.logL = -GMModels{numComponents}.NegativeLogLikelihood;