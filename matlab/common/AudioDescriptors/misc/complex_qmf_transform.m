function qmf_spectrum = complex_qmf_transform(x)

%if size(x,2) > size(x,1)
if size(x,1) > size(x,2)
    x = x';
end

M = length(x);

qmf_spectrum = zeros(M/2,1);

W = exp(1i * 2 * pi / M);

for k = 1:M/2
    qmf_spectrum(k) = sum( x .* ( W.^( (k-1+0.5) * ( 2*(0:(M-1)) + 1 ) ) ) );
end

%qmf_spectrum = sum( ( W.^( (remat(0:(M/2-1),[M/2,1])+0.5) * ( 2*(0:(M-1)) + 1 ) ) ) .*  x);