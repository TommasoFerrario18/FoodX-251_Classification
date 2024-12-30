close all
clear
clc
%% Datasets Load
disp("Load Labeled Train")
tic
train_imds = load_images("..\images\train_small.csv", "..\images\train_set\", DatasetType.train_20);
toc

disp("Load Unlabeled Train")
tic
train_un_imds = load_images("..\images\train_unlabeled.csv", "..\images\train_set\", DatasetType.train_un);
toc

%% Estrazione Feature
net = alexnet;
layer = "relu3";

[train_features, train_labels] = feature_extraction_NN(train_imds, net, layer, true, true);

%% Retrival
% https://it.mathworks.com/help/stats/knnsearch.html#d126e717833

%% Salvataggio risultati