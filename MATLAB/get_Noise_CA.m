function values = get_Noise_CA(option,length)

rng('shuffle'); rngState = rng;
seed = rngState.Seed + uint32(feature('getpid')); rng(seed);

if option == 1 
    rho = 0; 
    values = rand(length, 1); 
    values(values > rho) = 1;
    values(values <= rho) = 0;
elseif option == 2
    rho = 0.1; 
    values = rand(length, 1); 
    values(values > rho) = 1;
    values(values <= rho) = 0;
elseif option == 3
    rho = 0.5; 
    values = rand(length, 1); 
    values(values > rho) = 1;
    values(values <= rho) = 0;
elseif option == 4
    rho = 0.9; 
    values = rand(length, 1); 
    values(values > rho) = 1;
    values(values <= rho) = 0;

end
