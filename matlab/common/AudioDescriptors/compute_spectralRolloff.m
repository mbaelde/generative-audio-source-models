% Audio Spectrum Rolloff Frequency Estimator
% Example: AudioSpectrumRolloff = ASR('BaCl.mf.C4B4_3.wav',0.016);

% Description: Einai mia metriki toy se poso upsiles sixnotites sto fasma
% iparxei ena sigkekrimeno tmima tiw energeias.

function AudioSpectralRolloff = compute_spectralRolloff(freqspectrum,TH, f)

if size(freqspectrum,1) < size(freqspectrum,2)
    freqspectrum = freqspectrum';
end

%change of number-format
format long;

%calling the help-function frequency_spectrum
%freqspectrum = frequency_spectrum(audioData,frameperiod);
%freqspectrum = abs(spectrogram(audioData, 2*hopSize, hopSize, 2*hopSize));
[SampleNumPerFrame, ~]=size(freqspectrum);

% Right part initialization
tempSamplesSum = 0;

% Right Part of the equation calculation
for k=1:SampleNumPerFrame
    tempSamplesSum =  freqspectrum(k) + tempSamplesSum;
end
tempFramesSum = TH * tempSamplesSum;

% Left part initialization
tempSamplesSum2 = 0;

AudioSpectralRolloff = SampleNumPerFrame;

% Left Part of the equation calculation and checking the condition
for k=1:SampleNumPerFrame
    tempSamplesSum2 =  freqspectrum(k) + tempSamplesSum2;
    if tempSamplesSum2 < tempFramesSum
        AudioSpectralRolloff=k;
    end
end

AudioSpectralRolloff = f(AudioSpectralRolloff);