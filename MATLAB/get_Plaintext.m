function plaintext = get_Plaintext(taskID, num_replicate)

plaintext = floor(taskID / num_replicate);

if mod(taskID, num_replicate) ~= 0,
    plaintext = plaintext + 1;
end

end