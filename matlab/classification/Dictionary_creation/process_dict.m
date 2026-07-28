startup;

dico = 3;

%class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
%folder_clean = 'Clean database features/';
folder_clean = 'ESC-50/';
folder_interval = 'Interval/';

my_names = dir(data_folder);
my_names = my_names(3:end);
for n = 1:length(my_names)
    class{n} = my_names(n).name;
end

n_class = length(class);
%%
disp('Processing dictionnary...')
disp(['Dico ', num2str(dico)])

for n = 1:length(class)
    disp(['Currently: ',class{n}])
    clear aux_L mycdf feature aux_feature
    file = load(['Features/',folder_clean,folder_interval,'T',num2str(T(dico)),'/feature_',class{n},'.mat']);
    feature = file.feature;
    % Clear model
    for cnt = 1:length(feature)
        N_buffer = length(feature{cnt});
        count = 1;
        for b = 1:N_buffer
            if ~isempty(feature{cnt}{b}.spectrum_model)
                aux_feature{cnt}{count} = feature{cnt}{b};
                idx_empty{n}{cnt}{b} = 1;
                count = count + 1;
            else
                idx_empty{n}{cnt}{b} = 0;
            end
        end
    end
    feature = aux_feature;
    
%     freq = 0:fs/N_fft(dico):fs/2;
%     f = freq(1:N_spect(dico)+1);
%     feat_size = zeros(1,length(feature));
%     for i = 1:length(feature)
%         feat_size(i) = length(feature{i});
%     end
%     cnt = 1;
%     for b = 1:length(feature)
%         for i = 1:feat_size(b)
%             aux_L(cnt,:) = [log(diff(mixture_normcdf(f, feature{b}{i}.spectrum_model.mu, sqrt(feature{b}{i}.spectrum_model.sigma), feature{b}{i}.spectrum_model.mixing_coeff))),n,b];
%             cnt = cnt + 1;
%         end
%     end
%     save(['Features/',folder_clean,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_',class{n},'.mat'], 'aux_L','idx_empty')
end

aux_L = [];
for n = 1:length(class)
    aux_L_aux = load(['Features/',folder_clean,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_',class{n},'.mat']);
    aux_L = [aux_L; aux_L_aux.aux_L];
end
save(['Features/',folder_clean,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_T',num2str(T(dico)),'.mat'],'aux_L','idx_empty','-v7.3');