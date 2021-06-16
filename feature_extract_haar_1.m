function fea = feature_extract_haar_1(I)
    [m, n] = size(I);
    
    % calc integral image
    I = cumsum(cumsum(double(I),2));
    
    % Edge kernel: diagonal 4x4 px each;

    H = zeros(m-7,n-7);
    for i = 2:m-7
        for j = 2:n-7
            black = calc_intensity(I, i-1, j-1, i+3, j+3);
            white = calc_intensity(I, i+3, j-1, i+7, j+3);
            black2 = calc_intensity(I, i+3, j+3, i+7, j+7);
            white2 = calc_intensity(I, i+3, j-1, i+7, j+3);
            val = (black+black2) - (white+white2);
            H(i-1,j-1)=val;
        end
    end
    fea = double(H);
end

