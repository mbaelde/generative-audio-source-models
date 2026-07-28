N=10^6;

output = mixture_randn(N, moy, sig, prop);
out = mixture_normpdf(output, moy, sig, prop);

output = mixture_randn_mex(N, moy, sig, prop);
out = mixture_normpdf_mex(output, moy, sig, prop);
% for ii = 1:1000
%     mixture_normpdf(x_mc, mu_1, sigma_1, p_1);
%     mixture_normpdf_mex(x_mc, mu_1, sigma_1, p_1);
% end


%feature{part}.spectrum_model = univariate_spectrum_model(abs(spectrum).^2, fs, N_spect, M_set, f, verbose, criterion);


% for k = 1:n_class
%     prior_g(k) = sum(database_test(:,end-1) ==k);
% end
% k = 1;
% %for k = 1:n_class
%     disp(['Currently: class ', class{k}])
%     tic
%     L_bay{k} = identification_general(database_test, aux_L_training, prior_g, my_class, param);
%     elapsed_time{k} = toc;
% 
%end
%%
% A = randn(70000,400);
% %A = randn(70000,400,'gpuArray');
% tic
% for i  = 1:10
% B = sum(A.*A,1);
% end
% elapsed = toc
%%
% Y = zeros(1,n_dist);
% cnt = 1;
% for ii = 1:m-1
%     for jj = (ii+1):m
%         Y(cnt) = kl_variational_gmm_sym(model_test{ii}.spectrum_model.mixing_coeff', model_test{ii}.spectrum_model.mu', model_test{ii}.spectrum_model.sigma',...
%                                     model_test{jj}.spectrum_model.mixing_coeff', model_test{jj}.spectrum_model.mu', model_test{jj}.spectrum_model.sigma');
%         cnt = cnt + 1;
%         progressbar(cnt,n_dist)
%     end
% end

% tic
% if num_base == 1
%     [L_min, L_map, L_bay] = identification_general(database, aux_L, my_class, param);
% else
%     if strcmp(type,'Complex')
%         L_bay = identification_complex_discr(database_test, aux_L_training, my_class, param);
%     else
%         [L_min, L_map, L_bay] = identification_general(database_test, aux_L_training, my_class, param);
%     end
% end
% elapsed_time = toc;

%%
% parfor ii = 1:length(myfeature)
%     aux(ii) = sum(log(mixture_mvnpdf(data_to_reco, myfeature{ii}.mu, myfeature{ii}.Sigma, myfeature{ii}.ComponentProportion)));
% end
%%

%for k = 1:2
%     rmu(:,:,k) = repmat(mu(k,:),[N,1]);
%     sigma(:,:,k) = inv(sigma(:,:,k));
%     detsigma(k) = sqrt(det(sigma(:,:,k)));
% end
% [H] = histcn(samples,n_x-1,n_y-1,n_z-1);
% H_u = H(:);
% parfor i = 1:70000
% %     out = fast_mixture_mvnpdf(x, rmu, sigma, detsigma, p);
% %     mixture_mvnpdf(x,mu,sigma, p);
% likelihood_complete = sum(log(mixture_mvnpdf(samples,mu,sigma,p)));
% end
%
% aux = repmat(H_u',[70000,1]).*r_l_mcdf_u;
% likelihood = sum(aux,2);
%%

% %%
% for i = 1:30000
% %for m = 1:M
% weight(m) .* mvnpdf(x, mu(m,:), sigma(:,:,m));
% %end
% end
%%
% % value = 2.^(1:n_max);
%
% param.T = T(1);
% param.D = D;
% param.N_spect = N_spect(1);
% param.M_set = 1:20;
% param.criterion = criterion;
% param.verbose = verbose;
%
% n=1;
% %for i = 1:n_max
% %    i
% %    tic
%     [output,error] = compute_signal_feature(signal{n}, fs, param);
% %    elapse = toc;
% %    time_test_n(i,:) = [value(i), elapse];
% %end
%
% % 500 répétitions : 475 secondes
%
% % plot(time_test(:,1), time_test(:,2), '-+')
% % hold on
% % plot(time_test_n(:,1), time_test_n(:,2), '-+')
% %%
% parfor pp = 2:mu
%     %if b > (pp-1)
%         data = test_sound(1+(b-pp)*winsize:winsize*b);
%         [L{pp}, prior{pp}, L_min{pp}, L_map{pp}, norm_factor{pp}] = estimate_class(b, data,winsize,N_fft(pp), N_spect(pp), aux_L{pp}, L{pp}, prior{pp}, L_min{pp}, L_map{pp}, norm_factor{pp}, dict_size(pp),model_size{pp}, my_class(2:end));
%     %end
% end

%%
%[L_min, L_map, L_bay] = identification_general(database(1:1000,:), aux_L, my_class, param);
%%
% %feature = cell(1,n_samp);
% for i = 1:100
%     data = mydata(i,:);
%     spectrum = fft(data);
%     spectrum = spectrum(1:N_spect(dico));
%     spectrum_data = [real(spectrum)', imag(spectrum)'];
%     spectrum_norm = N_spect(dico) * spectrum_data ./ repmat(sum(abs(spectrum_data)),[N_spect(dico),1]);
%     data_to_model = [f', spectrum_norm];
%
%     BIC = zeros(1,length(M_set));
%     GMModels = cell(1,length(M_set));
%     options = statset('MaxIter',500);
%     for k = M_set
%         try
%             GMModels{k} = fitgmdist(data_to_model,k,'Options',options);
%             BIC(k)= GMModels{k}.BIC;
%         catch
%             GMModels{k} = [];
%             BIC(k) = inf;
%         end
%     end
%
%     [minBIC,numComponents] = min(BIC);
%
%     %feature{i} = GMModels{numComponents};
% end