close all
clear
clc

%% Load training set images
disp("Load Train")
tic
train_imds = load_images("..\images\train_small.csv", "..\images\train_set\", DatasetType.train_20);
toc

%% Setup AlexNet
net = alexnet;
sz = net.Layers(1).InputSize;
layer = "relu3";

%% Extract features - Training set
disp("Train Features")
tic
numImages = numel(train_imds.Files);
% Preallocate arrays
train_features = zeros(numImages, 384*13*13); % Size based on relu3 layer output
train_labels = zeros(numImages, 1);

% Process images in batches
batchSize = 50;
for i = 1:batchSize:numImages
    batchIdx = i:min(i+batchSize-1, numImages);
    
    % Read batch of images
    imgs = zeros(sz(1), sz(2), 3, length(batchIdx));
    for j = 1:length(batchIdx)
        imgs(:,:,:,j) = imresize(double(readimage(train_imds, batchIdx(j))), sz(1:2));
    end
    
    % Extract features for batch
    feats = activations(net, imgs, layer, "OutputAs", "rows");
    train_features(batchIdx,:) = feats;
    train_labels(batchIdx) = train_imds.Labels(batchIdx);
end
toc

%% Training
disp("Training")
tic
clf = fitcknn(train_features, train_labels, "Distance", "euclidean", "NumNeighbors", 3);
toc

%% Load and process validation set
disp("Load Validation")
tic
val_imds = load_images("..\images\val_info.csv", "..\images\val_set\", DatasetType.val);
toc

numValImages = numel(val_imds.Files);
test_features = zeros(numValImages, 384*13*13);
test_labels = zeros(numValImages, 1);

% Process validation images in batches
tic
for i = 1:batchSize:numValImages
    batchIdx = i:min(i+batchSize-1, numValImages);
    
    imgs = zeros(sz(1), sz(2), 3, length(batchIdx));
    for j = 1:length(batchIdx)
        imgs(:,:,:,j) = imresize(double(readimage(val_imds, batchIdx(j))), sz(1:2));
    end
    
    feats = activations(net, imgs, layer, "OutputAs", "rows");
    test_features(batchIdx,:) = feats;
    test_labels(batchIdx) = val_imds.Labels(batchIdx);
end
toc

%% Classification
disp("Classificazione test set")
tic
prediction = clf.predict(test_features);
toc

%% Performance
disp("Misure delle performance")
cm = confmat(test_labels, prediction);
show_confmat(cm.cm_raw, cm.labels, cm.accuracy);