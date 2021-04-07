function values = get_Noise(option,length)

% -------------------------------------------------------------------------
%   SEEDING NOISE
%       Gaussian distribution, mean 0.5, variance:
%
%       1 -- 0.5      2 -- 0.25      3 -- 0.1      4 -- no noise
% -------------------------------------------------------------------------

if option == 1
    sigma = 0.5;
    mu = 0.5;
    pd = makedist('Normal','mu',mu,'sigma',sigma);
    normal_trunc = truncate(pd, mu-sigma, mu+sigma);
    rng('shuffle'); rngState = rng;
    seed = rngState.Seed + uint32(feature('getpid')); rng(seed);
    values = random(normal_trunc, length, 1);
    
elseif option == 2
    sigma = 0.25;
    mu = 0.5;
    pd = makedist('Normal','mu',mu,'sigma',sigma);
    normal_trunc = truncate(pd, mu-sigma, mu+sigma);
    rng('shuffle'); rngState = rng;
    seed = rngState.Seed + uint32(feature('getpid')); rng(seed);
    values = random(normal_trunc, length, 1);
           
elseif option == 3
    sigma = 0.1;
    mu = 0.5;
    pd = makedist('Normal','mu',mu,'sigma',sigma);
    normal_trunc = truncate(pd, mu-sigma, mu+sigma);
    rng('shuffle'); rngState = rng;
    seed = rngState.Seed + uint32(feature('getpid')); rng(seed);
    values = random(normal_trunc, length, 1);
    
elseif option == 4
    rng('shuffle'); rngState = rng;
    seed = rngState.Seed + uint32(feature('getpid')); rng(seed);
    values = ones(length, 1) *0.5;
end 
end
