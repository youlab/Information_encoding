% Crop patterns into circular shape
function cropped_Img  = crop_Img(IM,x,y,scale, BACKGROUND)

idx = sqrt(x.^2 + y.^2) > (450/3/scale);
cropped_Img = IM;

if BACKGROUND == 0 % don't include plate
    cropped_Img(idx) = 0;
else % include plate
    cropped_Img(idx) = 1;
end

end