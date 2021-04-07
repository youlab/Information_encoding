function filterKernel = makeKernel(plateDiameter, gridSize,...
    positiveRadius, postitiveExponent, positiveBenefit, negativeRadius,...
    negativeExponent, maximumRepulsionRadius, pixelSize, kernelopt)

% make the dispersal kernel
% positiveRadius    - d1
% negativeRadius    - d2
% postitiveExponent - h1: inf
% negativeExponent  - h2: inf
% positiveBenefit   - b
% rKernel           - d
% vKernel           - f(d), formula 4 in Xavier's

rKernel = linspace(0, plateDiameter/2, gridSize); 
vKernel = positiveBenefit * 2.^(-(rKernel/positiveRadius).^postitiveExponent)- 2.^(-(rKernel/negativeRadius).^negativeExponent);


% make filter based on kernel
filterSize = floor(maximumRepulsionRadius * 2 / pixelSize);
if mod(filterSize, 2) == 0
    filterSize = filterSize + 1;
end

xFilter = ((0:(filterSize - 1)) - (filterSize/2)) + 0.5;
yFilter = xFilter;
[xFilter, yFilter] = meshgrid(xFilter, yFilter);
rFilter = sqrt(xFilter.^2 + yFilter.^2) * pixelSize;

% define the interaction kernel
% interpolation
filterKernel = interp1(rKernel, vKernel, rFilter, 'pchip');
 
% add white noise 
if kernelopt == 2
    spot = find(filterKernel > 0);
    a    = filterKernel(spot);
    snr  = 10; % signal-to-noise ratio
   
    rng('shuffle'); rngState = rng;
    seed = rngState.Seed + uint32(feature('getpid'));
    seed = cast(seed,'double');
    filterKernel(spot) = awgn(a,snr,'measured',seed);
    
elseif kernelopt == 3
    spot = find(filterKernel > 0);
    a    = filterKernel(spot);
    snr  = 3.5; 
   
    rng('shuffle'); rngState = rng;
    seed = rngState.Seed + uint32(feature('getpid'));
    seed = cast(seed,'double');
    filterKernel(spot) = awgn(a,snr,'measured',seed);
    
elseif kernelopt == 4
    spot = find(filterKernel > 0);
    a    = filterKernel(spot);
    snr  = 2; 
   
    rng('shuffle'); rngState = rng;
    seed = rngState.Seed + uint32(feature('getpid'));
    seed = cast(seed,'double');
    filterKernel(spot) = awgn(a,snr,'measured',seed);

end
end 


