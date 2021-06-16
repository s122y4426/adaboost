function fea = feature_extract_haar_4(I)
    [m, n] = size(I);
    
    % calc integral image
    I = cumsum(cumsum(double(I),2));
    
    % Edge kernel: (up + down) - center 2x6 px each;

    H = zeros(m-5,n-5);
    for i = 2:m-5
        for j = 2:n-5
            black = calc_intensity(I, i-1, j-1, i+1, j+5);
            white = calc_intensity(I, i+1, j-1, i+3, j+5);
            black2 = calc_intensity(I, i+3, j-1, i+5, j+5);
            val = black +black2 - white;
            H(i-1,j-1)=val;
        end
    end
    fea = double(H);
end

