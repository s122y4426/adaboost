function fea = feature_extract_haar_5(I)
    [m, n] = size(I);
    
    % calc integral image
    I = cumsum(cumsum(double(I),2));
    
    % Edge kernel: (left + right) - middle 6x2 px each;

    H = zeros(m-5,n-5);
    for i = 2:m-5
        for j = 2:n-5
            black = calc_intensity(I, i-1, j-1, i+5, j+1);
            white = calc_intensity(I, i-1, j+1, i+5, j+3);
            black2 = calc_intensity(I, i-1, j+3, i+5, j+5);
            val = black +black2 - white;
            H(i-1,j-1)=val;
        end
    end
    fea = double(H);
end

