function reduce_dictionnary(dico, T, N_fft, N_spect, fs, idx_m)

class = {'airplane','alarm','explosion','gunshot','helicopter','step','vehicule','voice_female','voice_male'};
folder_feat = 'Global database features/';
folder_clean = 'Clean database features/';
folder_interval = 'Interval/';

disp('Processing dictionnary...')
disp(['Dico ', num2str(dico)])

if isempty(idx_m) && 1
idx = { setdiff(1:19,[4,8,11,13,14,15,18]+1);
        1:6;
        1:30;
        setdiff(1:288,[31,32,98,101:105,125,127,130,131,133,134,173:180,183:192,194:196,214:221,228:232,242:245,280,281]+1);
        1:14;
        1:272;
        setdiff(1:84,[11,12,21,33,35,37,45,47,61,82]+1);
        1:9;
        1:53};
end

for n = 1:length(class);
    clear aux_L mycdf feature aux_feature
    file = load(['Features/',folder_feat,folder_interval,'T',num2str(T(dico)),'/feature_',class{n},'.mat']);
    feature = file.feature;
    % Clear model
    for cnt = 1:length(feature)
        N_buffer = length(feature{cnt});
        count = 1;
        for b = 1:N_buffer
            if ~isempty(feature{cnt}{b}.spectrum_model)
                aux_feature{cnt}{count} = feature{cnt}{b};
                count = count + 1;
            end
        end
    end
    feature = aux_feature;
    
    if isempty(idx_m) && 1
        % keep good sounds
        clear aux_feature
        cnt_2 = 1;
        for cnt = idx{n}
            N_buffer = length(feature{cnt});
            count = 1;
            for b = 1:N_buffer
                if ~isempty(feature{cnt}{b}.spectrum_model)
                    aux_feature{cnt_2}{count} = feature{cnt}{b};
                    count = count + 1;
                end
            end
            cnt_2 = cnt_2 + 1;
        end
        feature = aux_feature;
        save(['Features/',folder_clean,folder_interval,'T',num2str(T(dico)),'/feature_',class{n},'.mat'],'feature')
    end
    
    freq = 0:fs/N_fft(dico):fs/2;
    f = freq(1:N_spect(dico)+1);
    feat_size = zeros(1,length(feature));
    for i = 1:length(feature)
        feat_size(i) = length(feature{i});
    end
    min_feat = min(feat_size);
    id_feat = 1:length(feature);
    id = 1:length(feature);
    id_min = id(min_feat == feat_size);
    for b = 1:max(feat_size)
        mycdf = 0;
        if b > min_feat
            min_feat = min(feat_size(setdiff(id_feat,id_min)));
            id_feat = setdiff(id, id_min);
            id_min = [id_min, id(min_feat == feat_size)];
        end
        for i = id_feat
            mycdf = mycdf + mixture_normcdf(f, feature{i}{b}.spectrum_model.mu, sqrt(feature{i}{b}.spectrum_model.sigma), feature{i}{b}.spectrum_model.mixing_coeff);
        end
        mycdf = mycdf / length(id_feat);
        aux_L(b,:) = [log(diff(mycdf)), n, b];
        progressbar(b,max(feat_size))
    end
    save(['Features/',folder_clean,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_reduced_',class{n},'.mat'], 'aux_L')
end

aux_L = [];
for n = 1:length(class)
    aux_L_aux = load(['Features/',folder_clean,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_reduced_',class{n},'.mat']);
    aux_L = [aux_L; aux_L_aux.aux_L];
end
save(['Features/',folder_clean,folder_interval,'T',num2str(T(dico)),'/preprocess_dictionnary_reduced_T',num2str(T(dico)),'.mat'],'aux_L');