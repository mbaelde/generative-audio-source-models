function [feature_training, mean_training, std_training, coeff] = compute_descriptors(database_training, descriptors, param)

descriptor = descriptors.descriptor;

n_features = 0;
for d = 1:length(descriptor)
    n_features = n_features + cell2mat(descriptor{d}(2));
end

N = size(database_training,1);
feature_training = zeros(N,n_features);

if descriptors.verbose
    disp('Compute descriptors...')
end
fs = param.fs;
winsize = param.T;
nfft = param.T;
f = 0:fs/nfft:fs/2;
for n = 1:N
    data = database_training(n,1:end-2);
    
    fft_data = fft(data);
    amp_stft = abs(fft_data(1:nfft/2+1));
    
    cnt_features = 1;
    for d = 1:length(descriptor)
        if strcmp(descriptor{d}(1), 'Autocorrelation')
            feature_training(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = compute_AutoCorrelation(data,winsize);
            cnt_features = cnt_features + cell2mat(descriptor{d}(2));
        elseif strcmp(descriptor{d}(1), 'ZCR')
            feature_training(n,cnt_features) = compute_ZCR(data);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'Energy')
            feature_training(n,cnt_features) = compute_totalEnergy(amp_stft);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralCentroid')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            p_v = amp_stft / sum(amp_stft);
            feature_training(n,cnt_features) = compute_spectralCentroid(freq_v, p_v);
            if isnan(feature_training(n,cnt_features))
                feature_training(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralSpread')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            p_v = amp_stft / sum(amp_stft);
            feature_training(n,cnt_features) = compute_spectralSpread(freq_v, p_v, feature_training(n,cnt_features-1));
            if isnan(feature_training(n,cnt_features))
                feature_training(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralSkewness')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            p_v = amp_stft / sum(amp_stft);
            feature_training(n,cnt_features) = compute_spectralSkewness(freq_v, p_v, feature_training(n,cnt_features-2), feature_training(n,cnt_features-1));
            if isnan(feature_training(n,cnt_features))
                feature_training(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralKurtosis')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            p_v = amp_stft / sum(amp_stft);
            feature_training(n,cnt_features) = compute_spectralKurtosis(freq_v, p_v, feature_training(n,cnt_features-3), feature_training(n,cnt_features-2));
            if isnan(feature_training(n,cnt_features))
                feature_training(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralSlope')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            feature_training(n,cnt_features) = compute_spectralSlope(amp_stft,freq_v);
            if isnan(feature_training(n,cnt_features))
                feature_training(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralDecrease')
            feature_training(n,cnt_features) = compute_spectralDecrease(amp_stft');
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralRolloff')
            feature_training(n,cnt_features) = compute_spectralRolloff(amp_stft,0.95,f);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralFlatness')
            feature_training(n,cnt_features) = compute_spectralFlatness(amp_stft);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralCrest')
            feature_training(n,cnt_features) = compute_spectralCrest(amp_stft);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'MFCC')
            aux_desc = compute_MFCC(abs(fft_data),fs);
            feature_training(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = aux_desc(1:cell2mat(descriptor{d}(2)));
            cnt_features = cnt_features + cell2mat(descriptor{d}(2));
        elseif strcmp(descriptor{d}(1), 'f0')
            feature_training(n,cnt_features) = compute_f0(amp_stft, fs, winsize);
            harmonic_model = compute_harmonic_model(fft_data(1:nfft/2+1),fs, feature_training(n,cnt_features),20);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'HarmonicEnergy')
            feature_training(n,cnt_features) = compute_harmonicEnergy(harmonic_model.ampl);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'NoiseEnergy')
            feature_training(n,cnt_features) = compute_noiseEnergy(compute_totalEnergy(amp_stft), feature_training(n,cnt_features-1));
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'Noisiness')
            feature_training(n,cnt_features) = compute_noisiness(feature_training(n,cnt_features-1), compute_totalEnergy(amp_stft));
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'Tristimulus')
            feature_training(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = compute_tristimulus(harmonic_model);
            cnt_features = cnt_features + cell2mat(descriptor{d}(2));
        elseif strcmp(descriptor{d}(1), 'Inharmonicity')
            feature_training(n,cnt_features) = compute_inharmonicity(harmonic_model);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'HarmonicSpectralDeviation')
            feature_training(n,cnt_features) = compute_harmonicSpectralDeviation(harmonic_model);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'OEHarmonicEnergyRatio')
            feature_training(n,cnt_features) = compute_OEHarmonicEnergyRatio(harmonic_model);
            cnt_features = cnt_features + 1;
        end
    end
    if descriptors.verbose
        progressbar(n,N);
    end
end
training_class = database_training(:,end-1);
n_class = max(training_class);

if descriptors.delta == 1
    feature_training = [feature_training(1:end-1,:), diff(feature_training,1,1)];
    training_class = training_class(1:end-1);
elseif descriptors.delta == 2
    aux = diff(feature_training,1,1);
    feature_training = [feature_training(1:end-2,:), aux(1:end-1,:), diff(aux,1,1)];
    training_class = training_class(1:end-2);
end

if param.training
    pca_flag = descriptors.pca(1);
    pca_components = descriptors.pca(2);
    if pca_flag
        mean_training = mean(feature_training,1);
        std_training = std(feature_training,1);
        feature_training = (feature_training - repmat(mean_training, [length(feature_training),1])) ./ repmat(std_training, [length(feature_training),1]);
        disp('Dimension reduction')
        [coeff,score,~] = pca(feature_training, 'NumComponents', pca_components);
        feature_training = score;
    else
        mean_training = zeros(n_class,size(feature_training,2));
        std_training = zeros(n_class,size(feature_training,2));
        for k = 1:n_class
            mean_training(k,:) = mean(feature_training(training_class == k,:),1);
            std_training(k,:) = std(feature_training(training_class == k,:),1);
        end
        coeff = [];
    end
    idx = setdiff(1:size(feature_training,1),find(isnan(feature_training(:,1))));
    feature_training = [feature_training(idx,:), training_class(idx), ones(length(idx),1)];
else
    mean_training = [];
    std_training = [];
    coeff = [];
    idx = setdiff(1:size(feature_training,1),find(isnan(feature_training(:,1))));
    feature_training = feature_training(idx,:);
end

