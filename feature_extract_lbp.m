%function fea = feature_extract(I, type)
function fea = feature_extract_lbp(I)
    [m, n] = size(I);

    I = gray_image;
    I = cat(1, I(1, :), I, I(end, :));
    I = cat(2, I(:, 1), I, I(:, end));

    LBP = zeros(m, n);

    for i = 2:m+1
        for j = 2:n+1
            neighbor = I(i-1:i+1, j-1:j+1);
            neighbor(neighbor< neighbor(2, 2)) = 0;
            neighbor(neighbor>=neighbor(2, 2)) = 1;
            neighbor = [neighbor(1), neighbor(4), neighbor(7), neighbor(8), neighbor(9), neighbor(6), neighbor(3) ,neighbor(2)];
            n0 = bin2dec(num2str(neighbor));
            n1 = bin2dec(num2str(circshift(neighbor,-1, 2)));
            n2 = bin2dec(num2str(circshift(neighbor,-2, 2)));
            n3 = bin2dec(num2str(circshift(neighbor,-3, 2)));
            n4 = bin2dec(num2str(circshift(neighbor,-4, 2)));
            n5 = bin2dec(num2str(circshift(neighbor,-5, 2)));
            n6 = bin2dec(num2str(circshift(neighbor,-6, 2)));
            n7 = bin2dec(num2str(circshift(neighbor,-7, 2)));

            LBP(i-1, j-1) = min([n0 n1 n2 n3 n4 n5 n6 n7]);
        end
    end
    fea = LBP;
end

