function [feature,error] = compute_signal_feature(signal, fs, param, type)
% Extract parameters
T = param.T;
D = param.D;
N_spect = param.N_spect;
M_set = param.M_set;
criterion = param.criterion;
verbose = param.verbose;

signal_data = signal.signal;
signal_class = signal.class;
% Signal length
N_s = length(signal_data);
%t = (0:1/fs:(T-1)/fs)';
% FFT size
N_fft = T;
% Vector of frequencies
freq = 0:fs/N_fft:fs/2;
f = freq(1:N_spect+1)';
% Number of data when cutting the signal
N_part = floor((N_s - T) / D);
feature = cell(1,N_part);
% Parameters of STFT
%winsize = 1024;
%hopsize = winsize / 2;
%nfft = winsize;
%K = nfft/2+1;
error = cell(1);
cnt = 1;
for part = 1:N_part
    feature{part}.class = signal_class;
    % Current data to be processed
    data = signal_data((part-1)*D+1:(part-1)*D+T);
    
    % Pre-compute Fourier transform
    spectrum = fft(data);
    
    % Pre-compute spectrum for different frames
    %stft_data = spectrogram(data, winsize, hopsize, nfft);
    %amp_stft = abs(stft_data).^2;
    
    % Pre-compute energy envelop for different frames
    %env_energy_data = compute_energy_envelope(data, fs, 20);
    
    % ------------------------------------------------------------------ %
    % -------------------- MAIN FEATURES: SPECTRUM --------------------- %
    % ------------------------------------------------------------------ %
    % ---- Spectrum ---- %
    try
        if strcmp(type,'Interval')
            feature{part}.spectrum_model = univariate_spectrum_model(abs(spectrum).^2, fs, N_spect, M_set, f, verbose, criterion);
        elseif strcmp(type,'Non interval')
            feature{part}.spectrum_model = ninterval_spectrum_model(abs(spectrum).^2, N_spect, M_set, f, criterion);
        end
    catch
        feature{part}.spectrum_model = [];
        error{cnt} = num2str(part);
        cnt = cnt + 1;
    end
    %clc
%     % ------------------------------------------------------------------ %
%     % ----------------------- SPECTRAL FEATURES ------------------------ %
%     % ------------------------------------------------------------------ %
%     freq_v = (0:fs/nfft:(nfft-1)*fs/nfft)';
%     freq_v = freq_v(1:K);
%     p_v = abs(spectrum(1:K)) / sum(abs(spectrum(1:K)));
%     % ---- Audio Spectral Centroid ---- %
%     feature{part}.AudioSpectralCentroid = compute_spectralCentroid(freq_v, p_v);
%     
%     % ---- Audio Spectral Spread ---- %
%     feature{part}.AudioSpectralSpread = compute_spectralSpread(freq_v, p_v, feature{part}.AudioSpectralCentroid);
%     
%     % ---- Audio Spectral Skewness ---- %
%     feature{part}.AudioSpectralSkewness = compute_spectralSkewness(freq_v, p_v, feature{part}.AudioSpectralCentroid, feature{part}.AudioSpectralSpread);
%     
%     % ---- Audio Spectral Kurtosis ---- %
%     feature{part}.AudioSpectralKurtosis = compute_spectralKurtosis(freq_v, p_v, feature{part}.AudioSpectralCentroid, feature{part}.AudioSpectralSpread);
%     % ------------------------------------------------------ %
%     % ---- MODELIZATION BY PEARSON TYPE IV DISTRIBUTION ---- %
%     % ------------------------------------------------------ %
%     [a, lambda, m, nu] = moment2pearson(feature{part}.AudioSpectralCentroid, feature{part}.AudioSpectralSpread,...
%                                         feature{part}.AudioSpectralSkewness, feature{part}.AudioSpectralKurtosis);
%     
%     feature{part}.PearsonDensity = pearsonpdf(freq_v, lambda, a, m, nu, 'sum');
    
    %     % ------------------------------------------------------------------ %
    %     % -------------------- ENERGY ENVELOPE FEATURES -------------------- %
    %     % ------------------------------------------------------------------ %
    %     % ---- Attack time ---- %
    %     [feature{part}.AttackTime, feature{part}.AttackSlope, feature{part}.DecreaseSlope,...
    %         feature{part}.EffectiveDuration, feature{part}.EnergyModulation] = compute_attack_feature(env_energy_data, fs);
    %
    %     % ---- Log-Attack time ---- %
    %     feature{part}.LogAttackTime = compute_LAT(feature{part}.AttackTime);
    %
    %     % ---- Temporal Centroid ---- %
    %     feature{part}.TemporalCentroid = compute_temporalCentroid(env_energy_data, t);
    %
    %     % ---- Audio Spectral Variation ---- %
    %     feature{part}.AudioSpectralVariation = compute_spectralVariation(amp_stft);
    %
    %     % Divide the signal into smaller buffers
    %     N_buffer = size(stft_data,2);
    %     for buff = 1:N_buffer
    %         buffer = data(1+(buff-1)*hopsize:(buff-1)*hopsize+winsize);
    %         % ------------------------------------------------------------------ %
    %         % ---------------- INSTANTANEOUS TEMPORAL FEATURES ----------------- %
    %         % ------------------------------------------------------------------ %
    %         % ---- Auto Correlation ---- %
    %         feature{part}.ACcoefs(:,buff) = compute_AutoCorrelation(buffer,winsize);
    %
    %         % ---- Zero Crossing Rate ---- %
    %         feature{part}.ZCR(buff) = compute_ZCR(buffer);
    %
    %         % ------------------------------------------------------------------ %
    %         % ----------------------- SPECTRAL FEATURES ------------------------ %
    %         % ------------------------------------------------------------------ %
    %         % ---- Total spectral energy ---- %
    %         feature{part}.TotalEnergy(buff) = compute_totalEnergy(amp_stft(:,buff));
    %
    %
    %
    %         % ---- Audio Spectral Slope ---- %
    %         feature{part}.AudioSpectralSlope(buff) = compute_spectralSlope(amp_stft(:,buff), freq_v);
    %
    %         % ---- Audio Spectral Decrease ---- %
    %         feature{part}.AudioSpectralDecrease(buff) = compute_spectralDecrease(amp_stft(:,buff));
    %
    %         % ---- Audio Spectral Rolloff ---- %
    %         threshold = 0.95;
    %         feature{part}.AudioSpectralRolloff(buff) = compute_spectralRolloff(amp_stft(:,buff), threshold, f);
    %
    %         % ---- Audio Spectral Flatness ---- %
    %         feature{part}.AudioSpectralFlatness(buff) = compute_spectralFlatness(amp_stft(:,buff));
    %
    %         % ---- Audio Spectral Crest ---- %
    %         feature{part}.AudioSpectralCrest(buff) = compute_spectralCrest(amp_stft(:,buff));
    %
    %         % ---- MFCC ---- %
    %         feature{part}.MFCC(:,buff) = compute_MFCC(amp_stft(:,buff)',fs);
    %
    %         % ------------------------------------------------------------------ %
    %         % ----------------------- HARMONIC FEATURES ------------------------ %
    %         % ------------------------------------------------------------------ %
    %         % ---- Fundamental Frequency ---- %
    %         feature{part}.F0(buff) = compute_f0(abs(stft_data(:,buff)), fs, winsize);
    %
    %         % ---- Harmonic Model ---- %
    %         n_partials = 20;
    %         harmonic_model = compute_harmonic_model(stft_data(:,buff), fs, feature{part}.F0(buff), n_partials);
    %
    %         % ---- Harmonic Energy ----%
    %         feature{part}.HarmonicEnergy(buff) = compute_harmonicEnergy(harmonic_model.ampl);
    %
    %         % ---- Noise Energy ---- %
    %         feature{part}.NoiseEnergy(buff) = compute_noiseEnergy(feature{part}.TotalEnergy(buff), feature{part}.HarmonicEnergy(buff));
    %
    %         % ---- Noisiness ---- %
    %         feature{part}.Noisiness(buff) = compute_noisiness(feature{part}.NoiseEnergy(buff), feature{part}.TotalEnergy(buff));
    %
    %         % ---- Tristimulus ---- %
    %         feature{part}.Tristimulus(:,buff) = compute_tristimulus(harmonic_model);
    %
    %         % ---- Inharmonicity ---- %
    %         feature{part}.Inharmonicity(buff) = compute_inharmonicity(harmonic_model);
    %
    %         % ---- Harmonic spectral deviation ---- %
    %         feature{part}.HarmonicSpectralDeviation(buff) = compute_harmonicSpectralDeviation(harmonic_model);
    %
    %         % ---- Odd-to-even harmonic energy ratio ---- %
    %         feature{part}.OEHarmonicEnergyRatio(buff) = compute_OEHarmonicEnergyRatio(harmonic_model);
    %
    
    %     end
    %
    %     % ------------------------------------------------------------------ %
    %     % ------------------------- DELTA FEATURES ------------------------- %
    %     % ------------------------------------------------------------------ %
    %     feature{part}.DeltaACcoefs = diff(feature{part}.ACcoefs,1,2);
    %
    %     feature{part}.DeltaZCR = diff(feature{part}.ZCR);
    %
    %     feature{part}.DeltaTotalEnergy = diff(feature{part}.TotalEnergy);
    %
    %     feature{part}.DeltaAudioSpectralCentroid = diff(feature{part}.AudioSpectralCentroid);
    %
    %     feature{part}.DeltaAudioSpectralSpread = diff(feature{part}.AudioSpectralSpread);
    %
    %     feature{part}.DeltaAudioSpectralSkewness = diff(feature{part}.AudioSpectralSkewness);
    %
    %     feature{part}.DeltaAudioSpectralKurtusis = diff(feature{part}.AudioSpectralKurtosis);
    %
    %     feature{part}.DeltaAudioSpectralSlope = diff(feature{part}.AudioSpectralSlope);
    %
    %     feature{part}.DeltaAudioSpectralDecrease = diff(feature{part}.AudioSpectralDecrease);
    %
    %     feature{part}.DeltaAudioSpectralRolloff = diff(feature{part}.AudioSpectralRolloff);
    %
    %     feature{part}.DeltaAudioSpectralFlatness = diff(feature{part}.AudioSpectralFlatness);
    %
    %     feature{part}.DeltaAudioSpectralCrest = diff(feature{part}.AudioSpectralCrest);
    %
    %     feature{part}.DeltaMFCC = diff(feature{part}.MFCC,1,2);
    %
    %     feature{part}.DeltaDeltaMFCC = diff(feature{part}.DeltaMFCC,1,2);
    %
    %     feature{part}.DeltaF0 = diff(feature{part}.F0);
    %
    %     feature{part}.DeltaHarmonicEnergy = diff(feature{part}.HarmonicEnergy);
    %
    %     feature{part}.DeltaNoiseEnergy = diff(feature{part}.NoiseEnergy);
    %
    %     feature{part}.DeltaNoisiness = diff(feature{part}.Noisiness);
    %
    %     feature{part}.DeltaTristimulus = diff(feature{part}.Tristimulus,1,2);
    %
    %     feature{part}.DeltaInharmonicity = diff(feature{part}.Inharmonicity);
    %
    %     feature{part}.DeltaHarmonicSpectralDeviation = diff(feature{part}.HarmonicSpectralDeviation);
    %
    %     feature{part}.DeltaOEHarmonicEnergyRatio = diff(feature{part}.OEHarmonicEnergyRatio);
    %
    %progressbar(part,N_part);
end
