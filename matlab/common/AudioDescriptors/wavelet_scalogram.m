function [scalogram, C, L] = wavelet_scalogram(data, wname)

if size(data,1) > size(data,2)
    data = data';
end

N = length(data);

J = 9;
[C,L] = wavedec(data,J,wname);

scalogram = zeros(J,N);
for j = 1:J
    d = detcoef(C,L,j); 
    d = d(ones(1,2^j),:);
    scalogram(j,:) = wkeep(d(:)',N);
end

scalogram = scalogram(:);
I = find(abs(scalogram) < sqrt(eps));
scalogram(I) = zeros(size(I));
scalogram = reshape(scalogram,J,N);