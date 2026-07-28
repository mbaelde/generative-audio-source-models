function output = max_number_buffer(true_class, test_class, winsize, class)

N_buffer = floor(length(test_class) / winsize);

cnt_cl = 1;
for k = class
    cur_test_class = test_class(true_class == k);
    
    idx = cur_test_class == k;
    mmax = [];
    
    if isempty(idx)
        measure(cnt_cl) = 0;
    else
        cnt_max = 1;
        if idx(1)
            mmax = [mmax, idx(1)];
        end
        for i = 2:length(idx);
            if idx(i)
                mmax = [mmax, idx(i)];
            end
            if idx(i-1) ~= idx(i)
                num_max(cnt_max) = sum(mmax);
                mmax = [];
                cnt_max = cnt_max + 1;
            end
        end
        num_max(cnt_max) = sum(mmax);
        measure(cnt_cl) = max(num_max);
        num_max = 0;
    end
    cnt_cl = cnt_cl + 1;
end

output = round(measure / winsize) / N_buffer;
