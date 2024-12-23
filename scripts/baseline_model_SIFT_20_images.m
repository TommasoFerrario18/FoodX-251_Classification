close all
clear
clc

%% Load training set images 
tic
train_imds = load_images("..\images\train_small.csv", "..\images\train_set\", DatasetType.train_20);
toc

tic
train_un_imds = load_images("..\images\train_unlabeled.csv", "..\images\train_set\", DatasetType.train_un);
toc

tic
val_imds = load_images("..\images\val_info.csv", "..\images\val_set\", DatasetType.val);
toc

tic
val_deg_imds = load_images("..\images\val_info.csv", "..\images\val_set_degraded\", DatasetType.val_deg);
toc

