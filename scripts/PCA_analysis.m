% close all;
% clear
% clc

%% Datasets Load
disp("Load Labeled Train")
tic
train_imds = load_images("..\images\train_small.csv", "..\images\train_set\", DatasetType.train_20);
toc

%% Estrazione delle Features

%% PCA
data = train_features;

% Standardize the data
data_standardized = zscore(data);

% Perform PCA
[coeff, score, latent, tsquared, explained] = pca(data_standardized);

% Plot explained variance
figure;
bar(explained);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Explained Variance');