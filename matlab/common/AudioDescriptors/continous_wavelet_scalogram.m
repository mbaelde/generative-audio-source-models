function cwg = continous_wavelet_scalogram(data, scales, wname)

N = length(data);

max_sample = 50000;

N_part = ceil(N / max_sample);

N_scales = length(scales);
cwg = zeros(N_scales, N);

for part = 1:N_part
    if part * max_sample > N
        coeff = cwt(data(1+(part-1)*max_sample:end),scales,wname);
        cwg(:,1+(part-1)*max_sample:end) = N_scales * abs(coeff).^2 ./ repmat(sum(abs(coeff).^2), [N_scales,1]);
    else
        coeff = cwt(data(1+(part-1)*max_sample:part*max_sample),scales,wname);
        cwg(:,1+(part-1)*max_sample:part*max_sample) = N_scales * abs(coeff).^2 ./ repmat(sum(abs(coeff).^2), [N_scales,1]);
    end
    progressbar(part, N_part)
end