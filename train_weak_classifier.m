%function SVM_model = train_weak_classifier(feature_type, win_size, filename)

clc;
clear;
% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% load image set
[ids,gt]=textread(sprintf(VOCopts.imgsetpath,"dog_train"),'%s %d');


% Complete Task 2.1 - 2.4 here
train_num = 250;

ids = ids(1:train_num,:);
Y = gt(1:train_num,:);
disp("Start creating models")
disp("------------------------------------")
% create models
for i=1:5
    train_patch = strcat("X_train_",num2str(i));
    wc_name = strcat("WC_",num2str(i));
    disp(train_patch)
    X_train = {};
    for j=1:length(ids)
        I = imread(sprintf(VOCopts.imgpath,ids{j}));
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
        X_train{j,1} = I;
    end
    
    X_train = cell2mat(X_train);

    SVM_model = fitcecoc(X_train,Y);

    % Save the trained model
    save(wc_name,'SVM_model');
    disp("Model Saved")
    disp("------------------------------------")
end