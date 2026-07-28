function K = CalculKernel(pc,fERBs)

K = zeros( length(fERBs) , length(pc) );

for j = 1 : length(pc)
    
    n = fix( fERBs(end)/pc(j) - 0.75 ); % Number of harmonics
    k = zeros( size(fERBs) ); % Kernel
    
    % Normalize frequency w.r.t. candidate
    q = fERBs / pc(j);
    
    % Create kernel
    
    for i = [ 1 primes(n) ]
        a = abs( q - i );
        
        % Peak's weigth
        p = a < .25;
        k(p) = cos( 2*pi * q(p) );
        
        % Valleys' weights
        v = .25 < a & a < .75;
        k(v) = k(v) + cos( 2*pi * q(v) ) / 2;
        
    end
    
    % Apply envelope
    k = k .* sqrt( 1./fERBs  );
    
    % K+-normalize kernel
    k = k / norm( k(k>0) );
    
    K(:,j) = k;
    
end