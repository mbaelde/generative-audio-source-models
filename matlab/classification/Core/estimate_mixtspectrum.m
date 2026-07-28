function [est_spectrum,prop_em,likelihood, z] = estimate_mixtspectrum(feature_class, spectrum_test)

M = size(feature_class,1);
N = size(feature_class,2);
prop_em = ones(M,1) ./ M;

iter_max = 30;
likelihood = zeros(1,iter_max+1);
likelihood(1) = inf;
for iter = 1:iter_max
    % e-step
    z = repmat(prop_em,[1,N]) .* feature_class;
    z = z ./ repmat(sum(z),[M,1]);
    % m-step
    prop_em = sum(z .* repmat(spectrum_test,[M,1]),2) ./ sum(spectrum_test);
    % likelihood
    likelihood(iter+1) = sum( log( sum(repmat(prop_em,[1,N]) .* feature_class ) ) );
    if abs((likelihood(iter+1) - likelihood(iter)) / likelihood(iter+1)) < 1e-5;
        break;
    end
    %progressbar(iter,iter_max)
end
est_spectrum = sum(repmat(prop_em,[1,N]) .* feature_class );
%likelihood_end = likelihood(end);