close all;
clear;

% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% load image set for Adaboost
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,"dog_val"),'%s %d');

% load test image set for evaluation
[ids_test,gt_test]=textread(sprintf(VOCopts.imgsetpath,"csit5410_test"),'%s %d');

% Complete Task 3.1 - 3.3 here
WC_1 = load('WC_1.mat');
WC_2 = load('WC_2.mat');
WC_3 = load('WC_3.mat');
WC_4 = load('WC_4.mat');
WC_5 = load('WC_5.mat');

%attr_size = 100;
attr_size = length(ids);

ids_val = ids(1:attr_size,:);
Y_val = gt(1:attr_size,:);

%attr_size_test = 100;
attr_size_test = length(ids_test);

ids_test = ids_test(1:attr_size_test,:);
Y_test = gt_test(1:attr_size_test,:);

% initialize weights
W = ones(1, attr_size)*1/attr_size;
L = zeros(attr_size,5); %labels
E = zeros(1,5); % error
A = zeros(1,5); % alpha
A_checked = zeros(1,5); % checked_alpha_index

for l = 1:5 % determine alpha
    for i = 1:5 % number of models
        if l==1
            X_val = {};
            for k=1:length(ids_val) % get training images fit to the size
                I = imread(sprintf(VOCopts.imgpath,ids_val{k}));
                I = rgb2gray(I);
                I = histeq(I);
                I = imresize(I, [128 128]);
                if i==1
                    I = feature_extract_haar_1(I);
                elseif i==2
                    I = feature_extract_haar_2(I);
                elseif i==3
                    I = feature_extract_haar_3(I);
                elseif i==4
                    I = feature_extract_haar_4(I);
                elseif i==5
                    I = feature_extract_haar_5(I);
                end
                I = reshape(I, 1, []);
                X_val{k,1} = I;
            end
            
            X_val = cell2mat(X_val);

            if i==1 % predict only for the first time
                [label_1,score_1] = predict(WC_1.SVM_model,X_val);
                count = count_correctness(label_1, Y_val);
                fprintf("Correctness (Weak Classifier 1): %d / %d \n", count, attr_size);
                L(1:attr_size,i) = label_1;
            elseif i==2
                [label_2,score_2] = predict(WC_2.SVM_model,X_val);
                count_2 = count_correctness(label_2, Y_val);
                fprintf("Correctness (Weak Classifier 2): %d / %d \n", count_2, attr_size);
                L(1:attr_size,i) = label_2;
            elseif i==3
                [label_3,score_3] = predict(WC_3.SVM_model,X_val);
                count_3 = count_correctness(label_3, Y_val);
                fprintf("Correctness (Weak Classifier 3): %d / %d \n", count_3, attr_size);
                L(1:attr_size,i) = label_3;
            elseif i==4
                [label_4,score_4] = predict(WC_4.SVM_model,X_val);
                count_4 = count_correctness(label_4, Y_val);
                fprintf("Correctness (Weak Classifier 4): %d / %d \n", count_4, attr_size);
                L(1:attr_size,i) = label_4;
            elseif i==5
                [label_5,score_5] = predict(WC_5.SVM_model,X_val);
                count_5 = count_correctness(label_5, Y_val);
                fprintf("Correctness (Weak Classifier 5): %d / %d \n", count_5, attr_size);
                L(1:attr_size,i) = label_5;
            end
         end
    
        % calc errors
        err=0;
        if i ==1
         for j = 1:attr_size
            err = err + W(j)*(1-label_1(j)*Y_val(j))/2;
         end
        E(i) = err;
        elseif i==2
          for j = 1:attr_size
            err = err + W(j)*(1-label_2(j)*Y_val(j))/2;
          end
        E(i) = err;
        elseif i==3
         for j = 1:attr_size
            err = err + W(j)*(1-label_3(j)*Y_val(j))/2;
         end
        E(i) = err; 
        elseif i==4
          for j = 1:attr_size
            err = err + W(j)*(1-label_4(j)*Y_val(j))/2;
          end
        E(i) = err;  
        elseif i==5
         for j = 1:attr_size
            err = err + W(j)*(1-label_5(j)*Y_val(j))/2;
         end
        E(i) = err; 
        end
    end
    
    % find a wc with minimum error 
    idx_t = find(E==min(E(~A_checked)));
    idx = idx_t(1);
    if A_checked(idx) == 1
        idx = idx_t(2);
    end
    A_checked(idx) = 1;

    % update weights
    alpha = 0.5*log((1-E(idx))/E(idx));
    A(idx) = alpha;

    for i = 1:attr_size
        W(i) = W(i)*exp(-alpha*L(i,idx)*Y_val(i));
    end
    
    % normalize W
    for i = 1:size(W)
        W(i) = W(i)/sum(W);
    end
end
disp("-----------------------------------")
disp("Alpha values")
disp(A)
disp("-----------------------------------")

% Adaboost
disp("Creating a strong classifier...")
for i = 1:5 % number of models
    X_val = {};
    for k=1:length(ids_test) % get training images fit to the size
        I = imread(sprintf(VOCopts.imgpath,ids_test{k}));
        I = rgb2gray(I);
        I = histeq(I);
        I = imresize(I, [128 128]);
        if i==1
            I = feature_extract_haar_1(I);
        elseif i==2
            I = feature_extract_haar_2(I);
        elseif i==3
            I = feature_extract_haar_3(I);
        elseif i==4
            I = feature_extract_haar_4(I);
        elseif i==5
            I = feature_extract_haar_5(I);
        end
        I = reshape(I, 1, []);
        X_val{k,1} = I;
    end
    
    X_val = cell2mat(X_val);

    if i==1
        [label_1,score_1] = predict(WC_1.SVM_model,X_val);
    elseif i==2
        [label_2,score_2] = predict(WC_2.SVM_model,X_val);
    elseif i==3
        [label_3,score_3] = predict(WC_3.SVM_model,X_val);
    elseif i==4
        [label_4,score_4] = predict(WC_4.SVM_model,X_val);
    elseif i==5
        [label_5,score_5] = predict(WC_5.SVM_model,X_val);
    end
end

H = A(1)*label_1 + A(2)*label_2 + A(3)*label_3 + A(4)*label_4 + A(5)*label_5;

for n = 1:length(H)
    if H(n)>0
        H(n)=1;
    end
    H(n)=-1;
end

count_h = count_correctness(H, Y_test);
fprintf("Correctness (Strong Classifier): %d / %d \n", count_h, attr_size_test);


% Sliding window
disp("-----------------------------------")
disp("Detecting an object with sliding window ...")

inputs = [1,2,3];
for im = 1:length(inputs)
    fprintf("Working on image%d \n", im);
    img_name = append("test_images/",num2str(inputs(im)), '.jpg');
    img = imread(img_name);
    img = rgb2gray(img);
    [m, n] = size(img);
    
    %r90_img = rot90(img,2);
    
    T = {}; % images cliped by a sliding window
    stride_size = 10;
    idx_x = 1;
    for j = 1:stride_size:m-127
        idx_y = 1;
        for k = 1:stride_size:n-127
            img_t = img(j:j+127,k:k+127);
            T{idx_x,idx_y} = img_t;
            idx_y = idx_y+1;
        end
        idx_x = idx_x +1;
    end
    T_l = reshape(T,[],1);
    
    for i = 1:5 % number of models
         X_val = {};
        for k=1:length(T_l)
            I = T_l(k);
            I = cell2mat(I);
            I = imresize(I, [128 128]);
            if i==1
                I = feature_extract_haar_1(I);
            elseif i==2
                I = feature_extract_haar_2(I);
            elseif i==3
                I = feature_extract_haar_3(I);
            elseif i==4
                I = feature_extract_haar_4(I);
            elseif i==5
                I = feature_extract_haar_5(I);
            end
            I = reshape(I, 1, []);
            X_val{k,1} = I;
        end

        X_val = cell2mat(X_val);

        if i==1
            [label_1,score_1] = predict(WC_1.SVM_model,X_val);
        elseif i==2
            [label_2,score_2] = predict(WC_2.SVM_model,X_val);
        elseif i==3
            [label_3,score_3] = predict(WC_3.SVM_model,X_val);
        elseif i==4
            [label_4,score_4] = predict(WC_4.SVM_model,X_val);
        elseif i==5
            [label_5,score_5] = predict(WC_5.SVM_model,X_val);
        end
    end
    H = A(1)*score_1(:,2) + A(2)*score_2(:,2) + A(3)*score_3(:,2) + A(4)*score_4(:,2) + A(5)*score_5(:,2);
    
    figure();
    imshow(img);
    hold on;
    for i = 1:3
        v = maxk(H,3);
        cell = find(H==v(i));
        col = fix(cell/idx_x)+1;
        col = 1+(col-1)*10;
        row = mod(cell, idx_x);
        row = 1+(row-1)*10;
        rectangle('Position',[col,row,128,128],'Edgecolor', 'y', 'LineWidth',2);
    end
end

