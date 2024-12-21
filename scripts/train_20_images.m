close all;
clear
clc

%% Load training set images 

% Take images information from csv file
train_info = readtable("..\images\train_small.csv", "ReadVariableNames", false);
train_info.Properties.VariableNames = {'FileName', 'Label'};

fileNames = train_info{:, 'FileName'};

imageFolder = '..\images\train_set\';
allImages = dir(fullfile(imageFolder, '*.jpg'));
allFileNames = {allImages.name}';

filteredFileNames = intersect(allFileNames, fileNames, 'stable');
filteredFilePaths = fullfile(imageFolder, filteredFileNames);

train_imds = imageDatastore(filteredFilePaths);
train_imds.Labels = train_info{:, "Label"};