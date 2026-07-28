function [prop_em, alpha_b, L] = fit_mixture_dirichlet(x, param)

type = param.type;

if strcmp(type,'variational')
    M_em = param.M_em;
    u_0 = param.u_0;
    v_0 = param.v_0;
    M_em_init = M_em;
    N = size(x,1);
    D = size(x,2);
    
    stop = false;
    
    alpha_b = u_0 ./ v_0;
    E_alpha = psi(u_0) - log(v_0);
    
    prop_em = ones(M_em,1) ./ M_em;
    u_em = zeros(M_em,D);
    v_em = zeros(M_em,D);
    
    iter_max = param.iter_max;
    tol_prop = param.tol_prop;
    
    L = zeros(M_em,iter_max);
    cnt = 1;
    while ~stop || cnt > M_em_init
        %clc
        %disp(['cnt: ',num2str(cnt),' / ',num2str(M_em_init)])
        %idx_init = kmeans(sqrt(x/2),M_em);
        %r = full(ind2vec(idx_init'))';
        r = zeros(N,M_em);
        for n = 1:N
            r(n,randi(M_em)) = 1;
        end
        
        phi = zeros(M_em,D);
        nu = zeros(M_em,D);
        
        for iter = 1:iter_max
            %progressbar(iter,iter_max)
            % M-step
            prop_em = mean(r)';
            
            for j = 1:M_em
                for l = 1:D
                    set_lk = setdiff(1:D,l);
                    phi(j,l) = sum( r(:,j) .* alpha_b(j,l) .* (psi(sum(alpha_b(j,:))) - psi(alpha_b(j,l)) + sum( psi(1,alpha_b(j,set_lk)) .* alpha_b(j,set_lk) .* (E_alpha(j,set_lk) - log(alpha_b(j,set_lk))) ) ) );
                end
                nu(j,:) = sum( repmat(r(:,j),[1,D]) .* log(x) );
                
                if any(phi(j,:) < 0)
                    u_em(j,:) = u_0(j,:) - phi(j,:);
                else
                    u_em(j,:) = u_0(j,:) + phi(j,:);
                end
                v_em(j,:) = v_0(j,:) - nu(j,:);
            end
            
            % E-step
            alpha_b = u_em ./ v_em;
            ln_gammafrac = gammaln(sum(alpha_b,2)) - sum(gammaln(alpha_b),2);
            
            E_alpha = psi(u_em) - log(v_em);
            E_alpha2 = (psi(u_em) - log(u_em)).^2 + psi(1,u_em);
            
            aux_first = sum(alpha_b .* (repmat(psi(sum(alpha_b,2)),[1,D]) - psi(alpha_b)) .* (E_alpha - log(alpha_b)), 2);
            aux_second = 0.5 * sum((alpha_b.^2) .* (repmat(psi(1,sum(alpha_b,2)),[1,D]) - psi(1,alpha_b)) .* E_alpha2, 2);
            aux_third = zeros(M_em,1);
            for a = 1:D
                set_ab = setdiff(1:D,a);
                aux_third = aux_third + sum(repmat(alpha_b(:,a),[1,D-1]) .* alpha_b(:,set_ab) .* ( repmat(psi(1,sum(alpha_b,2)) .* (E_alpha(:,a) - log(alpha_b(:,a))),[1,D-1]) .* (E_alpha(:,set_ab) - log(alpha_b(:,set_ab))) ),2);
            end
            
            hat_R = ln_gammafrac + aux_first + aux_second + aux_third;
            rho = zeros(N,M_em);
            for j = 1:M_em
                rho(:,j) = log(prop_em(j)) + hat_R(j) + sum(repmat((alpha_b(j,:) - 1),[N,1]) .* log(x),2);
            end
            r = exp(LSE_normprob(rho));
            
            % Bound
            aux = zeros(1,5);
            for j = 1:M_em
                aux(1) = aux(1) + sum(r(:,j) .* (hat_R(j) + sum(repmat(alpha_b(j,:),[N,1]) .* log(x),2)));
                aux(3) = aux(3) + sum(u_0(j,:) .* log(v_0(j,:)) - gammaln(u_0(j,:)) + (u_0(j,:) - 1) .* E_alpha(j,:) - v_0(j,:) .* alpha_b(j,:));
                aux(5) = aux(5) + sum(u_em(j,:) .* log(v_em(j,:)) - gammaln(u_em(j,:)) + (u_em(j,:) - 1) .* E_alpha(j,:) - v_em(j,:) .* alpha_b(j,:));
            end
            aux_2 = r .* repmat(log(prop_em)',[N,1]); aux(2) = sum(aux_2(:));
            aux_4 = r .* log(r); aux(4) = sum(aux_4(:));
            L(cnt,iter) = sum(aux);
            if isnan(L(cnt,iter))
                break;
            end
        end
        
        idx = find(prop_em < tol_prop);
        if isempty(idx)
            stop = true;
        end
        new_idx = setdiff(1:M_em,idx);
        prop_em = prop_em(new_idx);
        u_em = u_em(new_idx,:);
        v_em = v_em(new_idx,:);
        u_0 = u_0(new_idx,:);
        v_0 = v_0(new_idx,:);
        M_em = M_em - length(idx);
        cnt = cnt + length(idx);
    end
elseif strcmp(type,'newton')
    M = param.M;
    N = size(x,1);
    D = size(x,2);
    
    prop_em = ones(M,1) ./ M;
    alpha_factor = param.alpha_factor;
    alpha_nr = alpha_factor*rand(M,D);
    
    iter_max = param.iter_max;
    L = zeros(1,iter_max);
    iter_nr_max = param.iter_nr_max;
    for iter = 1:iter_max
        disp(['iter: ',num2str(iter), ' / ',num2str(iter_max)])
        % E-step
        aux_dir = zeros(N,M); % log dirichlet proba
        for m = 1:M
            aux_dir(:,m) = dirichlet_pdf(x, alpha_nr(m,:));
        end
        z = exp(LSE_normprob( repmat(log(prop_em)',[N,1]) + aux_dir ));
        idx_nan = find(~isnan(z(:,1)));
        % M-step
        prop_em = mean(z(idx_nan,:))';
        for iter_nr = 1:iter_nr_max
            for m = 1:M
                Jf = diag(psi(1,alpha_nr(m,:))) - psi(1,sum(alpha_nr(m,:))) * ones(D,D);
                f = psi(alpha_nr(m,:)) - psi(sum(alpha_nr(m,:))) - sum(repmat(z(idx_nan,m),[1,D]) .* log(x(idx_nan,:)),1) ./ sum(z(idx_nan,m));
                alpha_nr(m,:) = alpha_nr(m,:) - (Jf \ f')';
            end
            alpha_nr(alpha_nr < 0) = rand(1);
        end
        if M == 1
            L(iter) = dirichlet_likelihood(x, alpha_nr);
        else
            L(iter) = mixture_dirichlet_likelihood(x, alpha_nr, prop_em);
        end
        if isnan(L(iter))
            L = L(1:iter);
            break;
        end
%         figure(1)
%         clf
%         plot(L(1:iter))
        %pause(0.001)
    end
    
    alpha_b = alpha_nr;
end