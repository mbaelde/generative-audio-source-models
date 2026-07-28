function ZeroCrossingRate = compute_ZCR(input)
%% compute_ZCR
%
% This function returns the zero crossing rate of an audio data.
%
% ZeroCrossingRate = compute_ZCR(input)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

% Substract the mean of the audio data
signal = input - mean(input);
% Get signal length
winLen = length(signal);

% Zero-Crossing Rate calculation
TempSignDifference = zeros(winLen-2,1);
for j=2:winLen-1
    TempSignDifference(j) = (1/(winLen-1)) * sum (abs(sign(signal(j)) - sign(signal(j-1))));
end
ZeroCrossingRate = sum(TempSignDifference);
