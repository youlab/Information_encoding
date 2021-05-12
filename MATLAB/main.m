% Mathematical model of bacterial pattern 
% Use this code to generate pattern dataset

close all
clear all
path = [pwd '/']; % root directory 

% PARAMETERS
% -------------------------------------------------------------------------
%    
%   SEEDING NOISE
%       1: Uniform distribution;              2: Gaussian distribution,large var;
%       3: Gaussian distribution, small var;  4: no noise
%
%   GROWTH KERNEL NOISE
%       1: no noise;             2: low white noise;
%       3: medium white noise;   4: high white noise
%
%   PLATE SHAPE
%       Default --    11: circule
%       Encryption -- 1: circle;  2:square; 3:diamond;  4:triangle
%       scale: shrink by scale
% 
%   TEXT LENGTH
%       2, 4, 6 or 8 bits
%
%   IFCROP
%       TRUE - for encryption only, otherwise FALSE
%
% -------------------------------------------------------------------------

r0 = 15; % radius of seeding configuration, 5
spacing = 50; % spot spacing, 15
seeding_noise = 2;
kernel_noise = 1;
plate_shape = 1;
scale = 1; % <=1
text_bit = 4; 
num_replicate = 1000; 

if plate_shape ~= 11
    IFCROP = 1; 
else 
    IFCROP = 0;
end

% SIMULATION 
% for i = 1:1%(2^text_bit-1)
%     for j = 1:num_replicate
%         id = (i - 1) * num_replicate + j;
%         encoding(id, path, r0, spacing, seeding_noise, kernel_noise, ...
%             plate_shape, scale, text_bit, num_replicate, IFCROP);
%     end 
% end

for i = 1:4
    encoding(i, path, r0, spacing, seeding_noise, kernel_noise, ...
             i, scale, text_bit, num_replicate, IFCROP);
end

% Save parameters
filename = strcat(path,'Simulation_Parameters.txt');
fileID = fopen(filename,'w');
fprintf(fileID, 'r0: %i\nspacing: %i\nseeding: %i\nkernel: %i\nshape: %i\nscale: %f\nbit: %i\ncrop: %i',...
    r0,spacing,seeding_noise,kernel_noise,plate_shape,scale,text_bit,IFCROP);
fclose(fileID);

disp('DONE')



