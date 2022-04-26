function pattern = ECA(patt, w, rule, time )

patt_c = patt + 1;
rule_arr = bitget(rule, 1:8) + 1;
pattern = zeros(n, width);

for i = 1:time
    
    pattern(i, :) = patt_c;   
    ind = sub2ind([2 2 2],[patt_c(2:end) patt_c(1)], patt_c, [patt_c(end) patt_c(1:end-1)]);
    patt_c = rule_arr(ind);
    
end

pattern = pattern-1;

end
