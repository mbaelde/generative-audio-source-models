function data_csv = spectrum2hist(spectrum, f_v)

rspectrum = round(spectrum);

count_f = [];
for i = 1:length(rspectrum)
    count_f = [count_f; f_v(i)*ones(rspectrum(i),1)];
end

data_csv = count_f;