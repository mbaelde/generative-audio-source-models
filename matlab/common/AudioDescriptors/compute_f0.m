function f0 = compute_f0(RF, fs, winsize)

dlog2p = 1/48;
dERBs = 0.1;
plim = [50;500];

log2pc = ( log2(plim(1)): dlog2p: log2(plim(2)) )';
pc = 2 .^ log2pc;

fERBs = erbs2hz((hz2erbs(min(pc)/4): dERBs: hz2erbs(fs/2))');

K = CalculKernel(pc,fERBs);

freq = (0:winsize/2)'*fs/winsize;

% Compute loudness at ERBs uniformly-spaced frequencies
L = sqrt( max( 0, interp1( freq, abs(RF(1:winsize/2+1)), fERBs, 'spline', 0) ) );

% Compute pitch strength
Si = pitchStrengthAllCandidates(L,K);

S = Si;

[~,i] = max(S);

if i == 1 || i == length(pc)
    f0 = pc(i);
else
    I = i-1 : i+1;
    tc = 1 ./ pc(I);
    ntc = ( tc/tc(2) - 1 ) * 2*pi;
    c = polyfit( ntc, S(I,1), 2 );
    ftc = 1 ./ 2.^( log2(pc(I(1))): 1/12/100: log2(pc(I(3))) );
    nftc = ( ftc/tc(2) - 1 ) * 2*pi;
    [~,k] = max( polyval( c, nftc ) );
    
    f0 = 2 ^ ( log2(pc(I(1))) + (k-1)/12/100 );
end

function erbs = hz2erbs(hz)
erbs = 6.44 * ( log2( 229 + hz ) - 7.84 );

function hz = erbs2hz(erbs)
hz = ( 2 .^ ( erbs./6.44 + 7.84) ) - 229;

function S = pitchStrengthAllCandidates(L,K)
% Normalize loudness
NL = L ./ sqrt( sum(L.*L) );

S = K' * NL;