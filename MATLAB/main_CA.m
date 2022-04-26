close all; clear all

path  = pwd;
path_text    = [path 'plaintext/'];
path_initial = [path 'initial/'];
path_final   = [path 'final/'];
path_image   = [path 'image/'];

mkdir(path_text)
mkdir(path_initial)
mkdir(path_final)
mkdir(path_image)

%-------------------------------------------------------
% ENCODING SETUP
rule = 60;
time = 500;
num_letter = 127;
num_bit = 7;
num_replicate = 1000;
spacing = 70; % max 70, min = 2*r
radius = 10; % radius of spots
noise_option = 3;
gridSize = 450; % sequence length


initial_matrix =[];
final_matrix = [];

for taskID = 1:num_letter*num_replicate
    
    %----------------------------------------
    % Get plaintext
    plaintext = get_Plaintext(taskID, num_replicate); % decimal
    plaintext_binary = plaintext2binary_bybit(plaintext,num_bit);
    
    % Get seeding centers
    [plaintext_length, num_bit] = size(plaintext_binary);
    num_seeding = sum(sum(plaintext_binary));
    
    l = size(plaintext_binary, 2);
    
    % Get seeding centers
    R = (l-1) / 2 * spacing;
    xcenter = linspace(-R, R, num_bit);    
    center  = zeros(num_seeding, 1);
    
    idx = 1;
    for i = 1:num_bit
        for j = 1: plaintext_length
            if(plaintext_binary(j, i) == 1)
                center(idx,1) = xcenter(i);
                idx = idx +1;
            end
        end
    end
    
    %----------------------------------------
    % Configuration field matrix, 0-> no seeding, 1-> seeding
    xx = ((0:(gridSize - 1)) - (gridSize/2)) + 0.5;
    
    for i = 1: size(center,1)
        x0 = center(i,1);
        idx = ((xx-x0).^2 - radius^2) < 0;
        if i == 1
            idx_list = idx;
        else
            idx_list = idx_list + idx;
        end
    end
    
    % Add noise
    config = double(idx_list);
    patt = double(idx_list);
    noise_length = sum(config);
    rng('shuffle')
    values = get_Noise_CA(noise_option, noise_length);
    k = find(config == 1);
    patt(k) = [values];
   
    % Evolution
    initial_matrix = [initial_matrix; patt];
    pattern = ECA(patt, length(patt), rule, time);
    final_matrix = [final_matrix; pattern(end,:)];
    
    % Save results
    fileID = fopen([path_final 'final_' num2str(taskID) '.txt'],'w');
    fprintf(fileID, '%i\n', pattern(end,:));
    fclose(fileID);
    
    fileID = fopen([path_initial 'initial_' num2str(taskID) '.txt'],'w');
    fprintf(fileID, '%i\n', pattern(1,:));
    fclose(fileID);
    
end

figure;imshow(initial_matrix)
figure;imshow(final_matrix)
