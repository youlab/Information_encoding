% Mathematical model of bacterial pattern 
% Use this code to generate pattern dataset

close all
clear all
path = [pwd '/']; % root directory 

% PARAMETERS
% -------------------------------------------------------------------------
%    
%   SEEDING NOISE
%       Gaussian distribution, mean 0.5, deviation:
%       1 -- 0.5      2 -- 0.25      3 -- 0.1      4 -- no noise
%
%   GROWTH KERNEL NOISE
%       1: no white noise;       2: low white noise;
%       3: medium white noise;   4: high white noise
%
%   PLATE SHAPE
%      Default -- 
%               11: circular plate
%      Encryption --
%               1: circle;  2:square;   3:rectangle;    
%               4:diamond;  5:triangle 
% 
%   TEXT LENGTH
%       2, 4, 6 or 8 bits
%
%   IFCROP
%       TRUE - for encryption only, otherwise FALSE
%
% -------------------------------------------------------------------------

r0 = 5; % radius of seeding configuration
spacing = 25; % spot spacing
seeding_noise = 2;
kernel_noise = 1;
plate_shape = 11;
text_bit = 4; 
num_replicate = 1000; 
IFCROP = 0; 


% SIMULATION
for i = 1: (2^text_bit-1)
    for j = 1:num_replicate
        id = (i - 1) * num_replicate + j;
        
        if id == 1 % Write simulation details in txt file 
            filename = strcat(path,'details.txt');
            fileID = fopen(filename,'w');
            formatSpec = 'r0:%i\nspacing:%i\nseeding_noise:%i\nkernel_noise:%i\nplate_shape:%i\ntext_bit:%i\nnum_replicate:%i\nIFCROP:%i\n';;
            fprintf(fileID,formatSpec,r0,spacing,seeding_noise,kernel_noise,plate_shape,text_bit,num_replicate,IFCROP);
            fclose(fileID);
        end

        encoding(id, path, r0, spacing, seeding_noise, kernel_noise, ...
            plate_shape, text_bit, num_replicate, IFCROP);
    end 
end

disp('DONE')



