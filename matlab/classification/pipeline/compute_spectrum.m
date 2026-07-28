function spectrum = compute_spectrum(database)

[N,T] = size(database(:,1:end-2));
F = T;
%spectrum = [fft(database(:,1:end-2),T,2), database(:,end-1:end)];

spectrum = zeros(N,F+2);
for n = 1:N
   spectrum(n,:) = [fft(database(n,1:end-2),F,2), database(n,end-1:end)];
end
