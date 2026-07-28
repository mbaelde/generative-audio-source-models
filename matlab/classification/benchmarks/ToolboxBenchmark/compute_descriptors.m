function [training_features, training_class, test_features, test_class] = compute_descriptors(database, descriptors, infos)

descriptor = descriptors.descriptor;

n_features = 0;
for d = 1:length(descriptor)
    n_features = n_features + cell2mat(descriptor{d}(2));
end

N = size(database.training,1);
training_features = zeros(N,n_features);

disp('Compute training descriptors...')
fs = infos.fs;
for n = 1:N
    data = database.training(n,1:end-2);
        
    winsize = 2^(nextpow2(infos.winsize * fs));
    nfft = winsize;
    
    fft_data = fft(data);
    amp_stft = abs(fft_data(1:nfft/2+1));
    
    f = 0:fs/nfft:fs/2;
    
    cnt_features = 1;
    for d = 1:length(descriptor)
        if strcmp(descriptor{d}(1), 'Autocorrelation')
            training_features(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = compute_AutoCorrelation(data,winsize);
            cnt_features = cnt_features + cell2mat(descriptor{d}(2));
        elseif strcmp(descriptor{d}(1), 'ZCR')
            training_features(n,cnt_features) = compute_ZCR(data);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'Energy')
            training_features(n,cnt_features) = compute_totalEnergy(amp_stft);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralCentroid')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            p_v = amp_stft / sum(amp_stft);
            training_features(n,cnt_features) = compute_spectralCentroid(freq_v, p_v);
            if isnan(training_features(n,cnt_features))
                training_features(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralSpread')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            p_v = amp_stft / sum(amp_stft);
            training_features(n,cnt_features) = compute_spectralSpread(freq_v, p_v, training_features(n,cnt_features-1));
            if isnan(training_features(n,cnt_features))
                training_features(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralSkewness')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            p_v = amp_stft / sum(amp_stft);
            training_features(n,cnt_features) = compute_spectralSkewness(freq_v, p_v, training_features(n,cnt_features-2), training_features(n,cnt_features-1));
            if isnan(training_features(n,cnt_features))
                training_features(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralKurtosis')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            p_v = amp_stft / sum(amp_stft);
            training_features(n,cnt_features) = compute_spectralKurtosis(freq_v, p_v, training_features(n,cnt_features-3), training_features(n,cnt_features-2));
            if isnan(training_features(n,cnt_features))
                training_features(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralSlope')
            freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
            freq_v = freq_v(1:nfft/2+1);
            training_features(n,cnt_features) = compute_spectralSlope(amp_stft,freq_v);
            if isnan(training_features(n,cnt_features))
                training_features(n,cnt_features) = 0;
            end
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralDecrease')
            training_features(n,cnt_features) = compute_spectralDecrease(amp_stft');
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralRolloff')
            training_features(n,cnt_features) = compute_spectralRolloff(amp_stft,0.95,f);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralFlatness')
            training_features(n,cnt_features) = compute_spectralFlatness(amp_stft);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'SpectralCrest')
            training_features(n,cnt_features) = compute_spectralCrest(amp_stft);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'MFCC')
            aux_desc = compute_MFCC(abs(fft_data),fs);
            training_features(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = aux_desc(1:cell2mat(descriptor{d}(2)));
            cnt_features = cnt_features + cell2mat(descriptor{d}(2));
        elseif strcmp(descriptor{d}(1), 'f0')
            training_features(n,cnt_features) = compute_f0(amp_stft, fs, winsize);
            harmonic_model = compute_harmonic_model(fft_data(1:nfft/2+1),fs, training_features(n,cnt_features),20);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'HarmonicEnergy')
            training_features(n,cnt_features) = compute_harmonicEnergy(harmonic_model.ampl);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'NoiseEnergy')
            training_features(n,cnt_features) = compute_noiseEnergy(compute_totalEnergy(amp_stft), training_features(n,cnt_features-1));
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'Noisiness')
            training_features(n,cnt_features) = compute_noisiness(training_features(n,cnt_features-1), compute_totalEnergy(amp_stft));
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'Tristimulus')
            training_features(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = compute_tristimulus(harmonic_model);
            cnt_features = cnt_features + cell2mat(descriptor{d}(2));
        elseif strcmp(descriptor{d}(1), 'Inharmonicity')
            training_features(n,cnt_features) = compute_inharmonicity(harmonic_model);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'HarmonicSpectralDeviation')
            training_features(n,cnt_features) = compute_harmonicSpectralDeviation(harmonic_model);
            cnt_features = cnt_features + 1;
        elseif strcmp(descriptor{d}(1), 'OEHarmonicEnergyRatio')
            training_features(n,cnt_features) = compute_OEHarmonicEnergyRatio(harmonic_model);
            cnt_features = cnt_features + 1;
        end
    end
    
    progressbar(n,N);
end
training_class = database.training(:,end-1);

if ~isempty(database.test)
    n_test_sounds = length(database.test);
    test_features = zeros(1,n_features);
    
    disp('Compute test descriptors...')
    for n = 1:n_test_sounds
        data = database.test(n,1:end-2);

        winsize = 2^(nextpow2(infos.winsize * fs));
        nfft = winsize;

        fft_data = fft(data);
        amp_stft = abs(fft_data(1:nfft/2+1));

        f = 0:fs/nfft:fs/2;

        cnt_features = 1;
        for d = 1:length(descriptor)
            if strcmp(descriptor{d}(1), 'Autocorrelation')
                test_features(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = compute_AutoCorrelation(data,winsize);
                cnt_features = cnt_features + cell2mat(descriptor{d}(2));
            elseif strcmp(descriptor{d}(1), 'ZCR')
                test_features(n,cnt_features) = compute_ZCR(data);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'Energy')
                test_features(n,cnt_features) = compute_totalEnergy(amp_stft);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralCentroid')
                freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
                freq_v = freq_v(1:nfft/2+1);
                p_v = amp_stft / sum(amp_stft);
                test_features(n,cnt_features) = compute_spectralCentroid(freq_v, p_v);
                if isnan(test_features(n,cnt_features))
                    test_features(n,cnt_features) = 0;
                end
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralSpread')
                freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
                freq_v = freq_v(1:nfft/2+1);
                p_v = amp_stft / sum(amp_stft);
                test_features(n,cnt_features) = compute_spectralSpread(freq_v, p_v, test_features(n,cnt_features-1));
                if isnan(test_features(n,cnt_features))
                    test_features(n,cnt_features) = 0;
                end
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralSkewness')
                freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
                freq_v = freq_v(1:nfft/2+1);
                p_v = amp_stft / sum(amp_stft);
                test_features(n,cnt_features) = compute_spectralSkewness(freq_v, p_v, test_features(n,cnt_features-2), test_features(n,cnt_features-1));
                if isnan(test_features(n,cnt_features))
                    test_features(n,cnt_features) = 0;
                end
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralKurtosis')
                freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
                freq_v = freq_v(1:nfft/2+1);
                p_v = amp_stft / sum(amp_stft);
                test_features(n,cnt_features) = compute_spectralKurtosis(freq_v, p_v, test_features(n,cnt_features-3), test_features(n,cnt_features-2));
                if isnan(test_features(n,cnt_features))
                    test_features(n,cnt_features) = 0;
                end
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralSlope')
                freq_v = (0:fs/nfft:(nfft-1)*fs/nfft);
                freq_v = freq_v(1:nfft/2+1);
                test_features(n,cnt_features) = compute_spectralSlope(amp_stft,freq_v);
                if isnan(test_features(n,cnt_features))
                    test_features(n,cnt_features) = 0;
                end
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralDecrease')
                test_features(n,cnt_features) = compute_spectralDecrease(amp_stft');
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralRolloff')
                test_features(n,cnt_features) = compute_spectralRolloff(amp_stft,0.95,f);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralFlatness')
                test_features(n,cnt_features) = compute_spectralFlatness(amp_stft);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'SpectralCrest')
                test_features(n,cnt_features) = compute_spectralCrest(amp_stft);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'MFCC')
                aux_desc = compute_MFCC(abs(fft_data),fs);
                test_features(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = aux_desc(1:cell2mat(descriptor{d}(2)));
                cnt_features = cnt_features + cell2mat(descriptor{d}(2));
            elseif strcmp(descriptor{d}(1), 'f0')
                test_features(n,cnt_features) = compute_f0(amp_stft, fs, winsize);
                harmonic_model = compute_harmonic_model(fft_data(1:nfft/2+1),fs, test_features(n,cnt_features),20);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'HarmonicEnergy')
                test_features(n,cnt_features) = compute_harmonicEnergy(harmonic_model.ampl);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'NoiseEnergy')
                test_features(n,cnt_features) = compute_noiseEnergy(compute_totalEnergy(amp_stft), test_features(n,cnt_features-1));
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'Noisiness')
                test_features(n,cnt_features) = compute_noisiness(test_features(n,cnt_features-1), compute_totalEnergy(amp_stft));
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'Tristimulus')
                test_features(n,cnt_features:cnt_features+cell2mat(descriptor{d}(2))-1) = compute_tristimulus(harmonic_model);
                cnt_features = cnt_features + cell2mat(descriptor{d}(2));
            elseif strcmp(descriptor{d}(1), 'Inharmonicity')
                test_features(n,cnt_features) = compute_inharmonicity(harmonic_model);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'HarmonicSpectralDeviation')
                test_features(n,cnt_features) = compute_harmonicSpectralDeviation(harmonic_model);
                cnt_features = cnt_features + 1;
            elseif strcmp(descriptor{d}(1), 'OEHarmonicEnergyRatio')
                test_features(n,cnt_features) = compute_OEHarmonicEnergyRatio(harmonic_model);
                cnt_features = cnt_features + 1;
            end
        end
        progressbar(n,n_test_sounds);
    end
    test_class = database.test(:,end-1);
end

if descriptors.delta == 1
    training_features = [training_features(1:end-1,:), diff(training_features,1,1)];
    training_class = training_class(1:end-1);
    if ~isempty(database.test)
        test_features = [test_features(1:end-1,:), diff(test_features,1,1)];
        test_class = test_class(1:end-1);
    end
elseif descriptors.delta == 2
    aux = diff(training_features,1,1);
    training_features = [training_features(1:end-2,:), aux(1:end-1,:), diff(aux,1,1)];
    training_class = training_class(1:end-2);
    if ~isempty(database.test)
        aux = diff(test_features,1,1);
        test_features = [test_features(1:end-2,:), aux(1:end-1,:), diff(aux,1,1)];
        test_class = test_class(1:end-2);
    end
end

pca_flag = descriptors.pca(1);
pca_components = descriptors.pca(2);
if pca_flag
    moy = mean(training_features,1);
    sig = std(training_features,1);
    training_features = (training_features - repmat(moy, [length(training_features),1])) ./ repmat(sig, [length(training_features),1]);
    if ~isempty(database.test)
        test_features = (test_features - repmat(moy, [length(test_features),1])) ./ repmat(sig, [length(test_features),1]);
    end
    disp('Dimension reduction')
    [coeff,score,~] = pca(training_features, 'NumComponents', pca_components);
    training_features = score;
    if ~isempty(database.test)
        test_features = test_features * coeff;
    end
end

idx = setdiff(1:length(training_features),find(isnan(training_features(:,1))));
training_features = training_features(idx,:);
training_class = training_class(idx);

if ~isempty(database.test)
    idx = setdiff(1:length(test_features),find(isnan(test_features(:,1))));
    test_features = test_features(idx,:);
    test_class = test_class(idx)';
else
    test_features = training_features;
    test_class = training_class;
end