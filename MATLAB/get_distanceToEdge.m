function distanceToEdge = get_distanceToEdge(plate)

distanceToEdge = bwdist(plate,'euclidean') * 250; 

end 

