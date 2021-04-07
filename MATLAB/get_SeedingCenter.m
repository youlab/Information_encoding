function center = get_SeedingCenter(plaintext_binary, spacing)

[plaintext_length, num_bit] = size(plaintext_binary);
num_seeding = sum(sum(plaintext_binary));
R = (num_bit-1)/2 * spacing;
xcenter = linspace(-R,R,num_bit);

if plaintext_length == 1
    ycenter = 0;
else
    ycenter = linspace(10,-10,plaintext_length);
end

% Get center locations
center  = zeros(num_seeding,2);
idx = 1;
for i = 1:num_bit
    for j = 1:plaintext_length
        if(plaintext_binary(j,i) == 1)
            center(idx,1) = xcenter(i);
            center(idx,2) = ycenter(j);
            idx = idx +1;
        end
    end
end
end
