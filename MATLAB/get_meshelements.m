function idx_list = get_meshelements(xx, yy, center, radius)

for i = 1: size(center,1),
    x0 = center(i,1);
    y0 = center(i,2);  
    idx = ((xx-x0).^2 + (yy-y0).^2 - radius^2) < 0;
    if i == 1,
        idx_list = idx;
    else
        idx_list = idx_list + idx;
    end
end

end