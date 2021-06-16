function fea = feature_extract_haar_2(I)
    [m, n] = size(I);
    
    % calc integral image
    I = cumsum(cumsum(double(I),2));
    
    % Edge kernel: up - down 4x4 px each;

    H = zeros(m-7,n-3);
    for i = 2:m-7
        for j = 2:n-3
            black = calc_intensity(I, i-1, j-1, i+3, j+3);
            white = calc_intensity(I, i+3, j-1, i+7, j+3);
            val = black - white;
            H(i-1,j-1)=val;
        end
    end
    fea = double(H);
end

