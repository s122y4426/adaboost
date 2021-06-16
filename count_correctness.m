function count = count_correctness(label, gt)
    count = 0;
    for i = 1:size(label,1)
        if label(i) == gt(i)
           count = count + 1;
        end
    end
end