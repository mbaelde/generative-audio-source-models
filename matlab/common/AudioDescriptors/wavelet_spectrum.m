function [wavelet_data, TIMES, FREQ] = wavelet_spectrum(data, Fs, winsize, hopsize, n_fft, wname)

[~,id] = min(fliplr(mod(50000,hopsize*(50:100))));
id = 100 - id;

max_sample = (hopsize)*(id+1);

N = length(data);

J = log2(n_fft/2);

if N > max_sample
    N_part = ceil(N / max_sample);
    
    cnt = 1;
    for part = 1:N_part
        if part*max_sample > N
            data_part = data((part-1)*max_sample:end);
        else
            data_part = data(1+(part-1)*max_sample:part*max_sample);
        end
        n_data_part = length(data_part);
        wpt = wpdec(data_part, J, wname);

        [spectrum,TIMES,FREQ] = wpspectrum(wpt,Fs);
        
        part_buffer = ceil((n_data_part) / hopsize) - 1;
        
        for b = 1:part_buffer
            if b*hopsize + winsize > n_data_part
                wavelet_data(:,cnt) = mean(flipud(spectrum(:,1+(b-1)*hopsize:end).^2),2);
            else
                wavelet_data(:,cnt) = mean(flipud(spectrum(:,1+(b-1)*hopsize:(b-1)*hopsize+winsize).^2),2);
            end
            cnt = cnt + 1;
        end
        progressbar(part,N_part)
    end
else
    N_buffer = floor(N / hopsize);
    wavelet_data = zeros(n_fft/2,N_buffer);

    wpt = wpdec(data, J, wname);

    [spectrum,TIMES,FREQ] = wpspectrum(wpt,Fs);

    for b = 1:N_buffer
        if b*hopsize + winsize > N
            wavelet_data(:,b) = mean(flipud(spectrum(:,1+(b-1)*hopsize:end).^2),2);
        else
            wavelet_data(:,b) = mean(flipud(spectrum(:,1+(b-1)*hopsize:(b-1)*hopsize+winsize).^2),2);
        end
    end
end