function [features, labels] = feature_extraction_NN(imds, net, layer, train, pca)
    numImages = numel(imds.Files);
    
    % Need to define sz before using it
    sz = [227 227]; % Common input size for many CNNs - adjust based on your network
    
    % Preallocate arrays for efficiency
    % Note: 384*13*13 is specific to relu3 layer output size - adjust if using different layer
    features = zeros(numImages, 384*13*13);
    labels = zeros(numImages, 1);
    
    % Process images in batches of 512 to manage memory usage
    batchSize = 512;
    tic
    for i = 1:batchSize:numImages
        batchIdx = i:min(i+batchSize-1, numImages);
        
        % Read and preprocess batch of images
        imgs = zeros(sz(1), sz(2), 3, length(batchIdx));
        for j = 1:length(batchIdx)
            imgs(:,:,:,j) = imresize(double(readimage(imds, batchIdx(j))), sz(1:2));
        end
        
        % Extract features for current batch
        feats = activations(net, imgs, layer, "OutputAs", "rows");
        features(batchIdx,:) = feats;
        
        % Fixed: Changed train_imds to imds
        labels(batchIdx) = imds.Labels(batchIdx);
    end
    
    if train && pca
        features = compute_pca(features, 150);
    elseif pca
        features = apply_pca(features, 150);
    end
    toc
end

function dataPCA = compute_pca(data, numberOfPrincipalComponent)
    % Standardize the data
    data_standardized = zscore(data);
    training_mean = mean(data); % Training mean
    training_std = std(data); % Training standard deviation
    
    % Perform PCA
    [coeff, score, ~, ~, ~] = pca(data_standardized);
    dataPCA = score(:, 1:numberOfPrincipalComponent);
    
    save('pca_info.mat', 'coeff', 'training_mean', 'training_std');
end

function dataPCA = apply_pca(data, numberOfPrincipalComponent)
    load('pca_info.mat', 'coeff', 'training_mean', 'training_std');
    test_data_standardized = (data - training_mean) ./ training_std;
    test_data_projected = test_data_standardized * coeff;
    dataPCA = test_data_projected(:, 1:numberOfPrincipalComponent);
end