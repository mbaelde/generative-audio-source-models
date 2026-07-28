function harmonic_model = compute_harmonic_model(stft_data, fs, f0, n_partials)

ampl = abs(stft_data);

S = [ampl; flipud(ampl(1:end-1))];

sins = spec_getsins_f0(S, fs, f0);
if sins(1,1) == 0
    sins = sins(:,2:end);
end
sins = sins(:,1:n_partials);

harmonic_model.f = sins(1,:);
harmonic_model.ampl = sins(2,:);
harmonic_model.phase = sins(3,:);
harmonic_model.f0 = f0;

% delta = -0.1:0.005:0.1;
% 
% for partial = 1:n_partials
%     cur_f = (f0 .* (1 + delta/log(1+partial))) .* partial;
%     L = zeros(3,length(delta));
%     idx = zeros(3,length(delta));
%     ampl_aux = zeros(length(delta),1);
%     for f_aux = 1:length(delta)
%         iidx = find(f_p - cur_f(f_aux) > 0 , 1);
%         iidx = [iidx-1, iidx];
%         if (ampl(iidx(1)) > ampl(iidx(2)))
%             idx(:,f_aux) = [iidx(1)-1,iidx];
%         else
%             idx(:,f_aux) = [iidx,iidx(2)+1];
%         end
%         L_0 = (cur_f(f_aux) - f_p(idx(2,f_aux))) * (cur_f(f_aux) - f_p(idx(3,f_aux))) / ((f_p(idx(1,f_aux)) - f_p(idx(2,f_aux))) * (f_p(idx(1,f_aux)) - f_p(idx(3,f_aux))));
%         L_1 = (cur_f(f_aux) - f_p(idx(1,f_aux))) * (cur_f(f_aux) - f_p(idx(3,f_aux))) / ((f_p(idx(2,f_aux)) - f_p(idx(1,f_aux))) * (f_p(idx(2,f_aux)) - f_p(idx(3,f_aux))));
%         L_2 = (cur_f(f_aux) - f_p(idx(1,f_aux))) * (cur_f(f_aux) - f_p(idx(2,f_aux))) / ((f_p(idx(3,f_aux)) - f_p(idx(1,f_aux))) * (f_p(idx(3,f_aux)) - f_p(idx(2,f_aux))));
%         L(:,f_aux) = [L_0;L_1;L_2];
%         ampl_aux(f_aux) = sum(ampl(idx(:,f_aux)) .* L(:,f_aux));
%     end
%     best_f = find(ampl_aux == max(ampl_aux));
%     harmonic_model.ampl(partial) = ampl_aux(best_f);
%     harmonic_model.phase(partial) = sum(phase(idx(:,best_f)) .* L(:,best_f));
%     harmonic_model.f(partial) = cur_f(best_f);
% end

% %%
% figure(1)
% clf
% plot(f_p(1:nfft/2+1), abs(stft_data(:,buff)))
% hold on
% for partial = 1:length(model.f)
%     plot(harmonic_model.f(partial),harmonic_model.ampl(partial), 'r*')
% end


