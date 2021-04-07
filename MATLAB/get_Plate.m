function plate = get_Plate(opt, pixelSize, plateDiameter, x, y, scale)

% -------------------------------------------------------------------------
%   PLATE SHAPE:
%       1: circular       2:square        3:diamond
%       4:triangle        5:rectangle     6:semisphere
%
%   SCALE:
%       shrink the plate area by scale       
% -------------------------------------------------------------------------
if opt == 11
    r = 225;
    plate  = sqrt(x.^2 + y.^2) >= r;
    
elseif opt == 1
    r = 254/scale;
   % r = 225; % for normal circular simulation
    plate  = sqrt(x.^2 + y.^2) >= r; 
 
elseif opt == 2
    l = 450/scale;
    a = abs(x) < (l/2);
    b = abs(y) < (l/2);
    plate = ~(a & b);

elseif opt == 3,
    l = 450/scale;
    k =  floor(l/sqrt(2));
    a = (x < y + k);
    b = (x < -y + k);
    c = (x > -y - k);
    d = (x > y - k);
    plate = ~(a & b & c & d);
    
elseif opt == 4,
    R = 684/scale;
    h = R/2/sqrt(3);
    k = R/sqrt(3);
    a = (x < (y+ k)/sqrt(3));
    b = (x > -(y + k)/sqrt(3));
    c = (y <  h);
    plate = ~(a & b & c);
    
elseif opt == 5
    half_width  = 0.4 * plateDiameter/pixelSize/2;
    half_length = 1 * plateDiameter/pixelSize/2;
    a = abs(x) < half_width;
    b = abs(y) < half_length;
    plate = ~(a & b);

elseif opt == 6,
    yy = y - 225/3;
    a = (y < 225/3);
    b = sqrt(x.^2 + yy.^2)*pixelSize >= plateDiameter/2; 
    c = (y < 225*2/3);
    plate = ~(~(a & b ) & c);
end

end
