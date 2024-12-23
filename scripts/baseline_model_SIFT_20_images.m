close all
clear
clc

%% Load training set images 
tic
train_imds = load_images("..\images\train_small.csv", "..\images\train_set\", DatasetType.train_20);
toc

% tic
% train_un_imds = load_images("..\images\train_unlabeled.csv", "..\images\train_set\", DatasetType.train_un);
% toc
% 

% 
% tic
% val_deg_imds = load_images("..\images\val_info.csv", "..\images\val_set_degraded\", DatasetType.val_deg);
% toc

%% Extract SIFT features
tic
[SIFT_fe, SIFT_le, C] = extract_SIFT_feature_train(5, 60, train_imds, 50);
toc

%% Training the model
tic
clf = fitcknn(SIFT_fe, SIFT_le, "Distance","euclidean", "NumNeighbors", 3);
toc

%% Load Validation set
tic
val_imds = load_images("..\images\val_info.csv", "..\images\val_set\", DatasetType.val);
toc

%% Istogrammi Test
disp("Rappresentazione BOW Test")
tic
[SIFT_fe_test, SIFT_le_test] = extract_SIFT_feature_test(5, 60, val_imds, 50, C);
toc

%% %% Classificazione del test
disp("Classificazione test set")
prediction = clf.predict(SIFT_fe_test);

%% Misure di performance
disp("Misure delle performance")
cm = confmat(SIFT_le_test, prediction);
show_confmat(cm.cm_raw, cm.labels, cm.accuracy);