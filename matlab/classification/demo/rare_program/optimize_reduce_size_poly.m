function N_sounds = optimize_reduce_size_poly(feature_training, Z, database, m, th, param)

id_class = unique(feature_training(:,end-1));
n_class = length(id_class);

true_class = database(:,end-1);
id_class = unique(feature_training(:,end-1));
prior_g = zeros(1,n_class);
for k = 1:n_class
    prior_g(k) = sum(database(:,end-1) == id_class(k));
end
gm = param.gm;
th_value = th.value;
th_type = th.type;

if strcmp(th_type,'f1')
    %%
    reduce_size = [20,10,5,1];
    %[175, 46, 13, 114, 30, 18, 40]
    %[57  10   1  34   8   1  10]
    N_sounds = th.initial_guess;
    %%
    iter = 1;
    stop = false;
    a_stop = zeros(1,n_class);
    while ~stop
        N_sounds_old = N_sounds;
        feature_norm = reduce_dictionary(feature_training, Z, N_sounds);
        
        [posterior_g,computation_time] = identification(database, feature_norm, prior_g, param);
             
        N = size(posterior_g,1);
        labels = zeros(1,N);
        for b = 1:N
            if mod(b, m) == 0
                sum_g = sum(gather(posterior_g((b-m+1):b,:)),1,'omitnan');
                if sum(sum_g) == 0
                    labels((b-m+1):b) = 0;
                else
                    labels((b-m+1):b) = find(max(sum_g) == sum_g);
                end
            end
        end

        true_class = database(:,end-1)';
        idx_zero = true_class == 0;
        true_class = true_class(~idx_zero);
        labels = labels(~idx_zero);
        idx_zero = labels == 0;
        true_class = true_class(~idx_zero);
        labels = labels(~idx_zero);

        N = length(labels);

        matrix_true = zeros(3, N);
        matrix_pred = zeros(3, N);
        for n = 1:N
            if true_class(n) == 0
                matrix_true(:,n) = 0;
            else
                matrix_true(unique(gm(true_class(n),:)),n) = 1;
            end
                matrix_pred(unique(gm(labels(n),:)),n) = 1;
            %progressbar(n,N)
        end

        % Segment-based metrics
        for k = 1:n_class
            [f1_score(k), error_rate(k)] = metrics_sed(matrix_true(:,true_class == id_class(k)), matrix_pred(:,true_class == id_class(k)));
        end
        f1_score = f1_score*100;
        error_rate = error_rate*100;
        
        for k = 1:n_class
            if f1_score(k) < th_value
                if N_sounds(k) == 1
                    a_stop(k) = true;
                else
                    a_stop(k) = false;
                    if N_sounds(k) <= 20
                        N_sounds(k) = N_sounds(k) - 1;
                    elseif iter < 5
                        N_sounds(k) = N_sounds(k) - reduce_size(1);
                    elseif iter >= 5 && iter < 20
                        N_sounds(k) = N_sounds(k) - reduce_size(2);
                    elseif iter >= 20 && iter < 50
                        N_sounds(k) = N_sounds(k) - reduce_size(3);
                    else
                        N_sounds(k) = N_sounds(k) - reduce_size(4);
                    end
                end
            elseif f1_score(k) > th_value
                N_sounds(k) = N_sounds(k) + 1;
                a_stop(k) = true;
%             else
%                 a_stop(k) = true;
            end
        end
        stop = all(a_stop) || iter > 100;
        iter = iter + 1;
        disp('-----------')
        disp(['iter: ',num2str(iter)])
        disp(['f1: ',num2str(f1_score)])
        disp(['mean f1: ',num2str(mean(f1_score))])
        disp(['N_sounds: ',num2str(N_sounds)])
        disp(['computation time: ',num2str(mean(computation_time)*1e3),'ms'])
        disp('-----------')
    end
elseif strcmp(th_type,'time')
    reduce_size = [20,10,5,1];
    N_sounds = [200,150,90,100,100];
    
    iter = 1;
    stop = false;
    a_stop = false;
    while ~stop
        N_sounds_old = N_sounds;
        feature_norm = reduce_dictionary(feature_training, Z, N_sounds);
        
        [~,computation_time] = identification(database, feature_norm, prior_g, param);
        mean_time = mean(computation_time);
        if mean_time > th_value
            if N_sounds == 1
                a_stop = true;
            else
                a_stop = false;
                if N_sounds <= 20
                    N_sounds = N_sounds - 1;
                elseif iter < 5
                    N_sounds = N_sounds - reduce_size(1);
                elseif iter >= 5 && iter < 20
                    N_sounds = N_sounds - reduce_size(2);
                elseif iter >= 20 && iter < 50
                    N_sounds = N_sounds - reduce_size(3);
                else
                    N_sounds = N_sounds - reduce_size(4);
                end
            end
        else
            a_stop = true;
        end
        stop = all(a_stop) || iter > 100;
        iter = iter + 1;
        disp('-----------')
        disp(['iter: ',num2str(iter)])
        disp(['time: ',num2str(mean_time)])
        disp(['N_sounds: ',num2str(N_sounds)])
        disp('-----------')
    end
end

N_sounds = N_sounds_old;
