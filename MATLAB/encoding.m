function encoding(taskID,path,r0,spacing, seeding_noise, kernel_noise, ...
    plate_shape, scale, text_bit, num_replicate, IFCROP)

% Paths
path_text    = [path 'plaintext/'];
path_initial = [path 'initial/'];
path_final   = [path 'final/'];

mkdir(path_text)
mkdir(path_initial)
mkdir(path_final)

if IFCROP == 1
    path_cropped = [path 'cropped/'];
    mkdir(path_cropped)
end

%-------------------------------------------------------------
% PARAMETERS
d1d2              = 0.4; % d1/d2
positiveBenefit   = 6.5; % b
positiveRadius    = 1000; % d1
negativeRadius    = positiveRadius/d1d2; % d2
postitiveExponent = 2000; % h1
negativeExponent  = postitiveExponent; % h2
plateRadius       = negativeRadius;
plateExponent     = 2000;
pixelSize         = 200;% resolution, um per pixel

if IFCROP == 0
    plateDiameter = pixelSize * 450;% Petri dish diameter, um
elseif IFCROP == 1
    plateDiameter = pixelSize * 790;
end
nt = 3000;
%% -------------------------------------------------------------
% GROWTH DOMAIN

% Define grid
maximumRepulsionRadius = negativeRadius*(log2(10^(1.2)).^(1/negativeExponent));
gridSize = plateDiameter/pixelSize;
if mod(gridSize, 2) == 0
    gridSize = gridSize + 1;
end

% Define plate
x = ((0:(gridSize - 1)) - (gridSize/2)) + 0.5;
y = x;
[x, y] = meshgrid(x,y);
plate = get_Plate(plate_shape,x,y,scale);
figname = strcat(path_initial, 'Shape_ID_', num2str(taskID),'.jpg');
imwrite(plate,figname,'jpg');

% Get distance of plate center to edge
r = sqrt(x.^2 + y.^2)*pixelSize;
distanceToEdge = get_distanceToEdge(plate); 

%% -------------------------------------------------------------
% PLAINTEXT
plaintext = get_Plaintext(taskID, num_replicate); 
plaintext_binary = plaintext_de2bi(plaintext,text_bit);

% Save plaintext
textname = strcat(path_text,'Plaintext_ID_', num2str(taskID),'.txt');
fileID = fopen(textname,'w');
fprintf(fileID, '%i', plaintext);
fclose(fileID);

%% -------------------------------------------------------------
% INITIAL PATTERN
center = get_SeedingCenter(plaintext_binary, spacing);
% center = [0 0]

% Configuration, 0-> no seeding, 1-> spot seeding
config = get_meshelements(x, y, center, r0);

% Add noise
noise_length = sum(sum(config));
values = get_Noise(seeding_noise,noise_length);
colony = zeros(gridSize);

% Plot seeding configuration
colony_copy = plate + colony;
colony_copy(config == 1) = 1;
figname = strcat(path_initial, 'Config_ID_', num2str(taskID),'.jpg');
imwrite(colony_copy,figname,'jpg');

colony(config == 1) = values;

% Plot initial pattern
figname1 = strcat(path_initial, 'IImg_ID_', num2str(taskID),'.jpg');
imshow(colony);
imwrite(colony, figname1,'jpg');

%% -------------------------------------------------------------
% GROWTH
colonyTracks = colony;
iterationResidual = nan(nt, 1);

figure
for i = 1:nt
    colonyOld = colonyTracks;
    filterKernel =...
        makeKernel(plateDiameter, gridSize, positiveRadius,...
        postitiveExponent, positiveBenefit, negativeRadius,...
        negativeExponent, maximumRepulsionRadius, pixelSize, kernel_noise);
    growth = filter2(filterKernel, colony);
    
    % Substract plate influence
    plateInfluence = -2.^(-(distanceToEdge/plateRadius).^(plateExponent))*1000; 
    plateInfluence(plate) = 0;
    growth = growth + plateInfluence; 
    
    % Update colony
    growth(plate) = 0;
    colony = colony + growth;
    colony(colony < 0) = 0;
    mVal = 8;
    colony(colony > mVal) = mVal;
    colonyTracks(colony > 0.1) = 1;
    
    % Plot colony
    IM = colonyTracks*0 + colony + plate*5;
    if mod(i,10) == 0
        imshow(IM)
        figname = strcat(path_final,int2str(taskID),'_',int2str(i),'.jpg');
        imwrite(IM, figname, 'jpg');
    end
    
    % Terminate the simulation if colony stops growing
    iterationResidual(i) = length(find(colonyOld ~= colonyTracks));
    if iterationResidual(i) == 0
        break
    end
    
end

% Save final pattern
figname2 = strcat(path_final, 'FImg_ID_', num2str(taskID),'.jpg');
imwrite(IM, figname2, 'jpg');

% Save cropped pattern center
if IFCROP == 1
    cropped_Img = crop_Img(IM,x,y,scale,0); 
    cropped_Img_copy = crop_Img(IM,x,y,scale,1);    
    a = (plateDiameter/pixelSize-450/scale)/2;
    b = a + 450/scale;
    cropped_Img = cropped_Img(a:b,a:b);
    cropped_Img_copy = cropped_Img_copy(a:b,a:b);
    size(cropped_Img)
    
    figure;imshow(cropped_Img);
    figname3 = strcat(path_cropped, 'Cropped_ID_', num2str(taskID),'.jpg');
    imwrite(cropped_Img, figname3, 'jpg');
    figname4 = strcat(path_cropped, 'Cropped_WBG_ID_', num2str(taskID),'.jpg');
    imwrite(cropped_Img_copy, figname4, 'jpg');

end
end
