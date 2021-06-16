function fea = feature_extract_haar_3(I)
    [m, n] = size(I);
    
    % calc integral image
    I = cumsum(cumsum(double(I),2));
    
    % Edge kernel: left - right 4x4 px each;

    H = zeros(m-3,n-7);
    for i = 2:m-3
        for j = 2:n-7
            black = calc_intensity(I, i-1, j-1, i+3, j+3);
            white = calc_intensity(I, i-1, j+3, i+3, j+7);
            val = black - white;
            H(i-1,j-1)=val;
        end
    end
    fea = double(H);
end

