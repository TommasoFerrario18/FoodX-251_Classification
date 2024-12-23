function [features, labels, C] = extract_SIFT_feature_train(featStep, imsize, imds, K)
    % Main function to extract SIFT features and create Bag of Words representation
    % Input:
    %   featStep: Step size for grid generation
    %   imsize: Target size for image resizing
    %   imds: ImageDatastore containing input images
    %   K: Number of clusters for k-means
    % Output:
    %   features: Final BoW representation
    %   labels: Corresponding image labels
    %   C: Cluster centroids from k-means

    % Construction of the grid on which SIFT descriptors will be extracted
    disp('Grid construction')
    pointPositions = generate_grid(featStep, imsize);
    
    % Estrazione features sul training
    disp('estrazione features')
    [features, labels] = extract_SIFT_imds(imds, imsize, pointPositions);
    
    % Creazione vocabolario
    disp('kmeans')
    [IDX, C] = kmeans(features, K, "MaxIter", 100, "Distance", "sqeuclidean", ...
            'Options', statset('UseParallel', true));
    
    % Creazione BoW per training
    disp('rappresentazione BOW training')
    [features, labels] = create_bow(IDX, labels, imds, K);
end

function pointPositions = generate_grid(featStep, imsize)
    % Generate regular grid of points for SIFT feature extraction
    % Input:
    %   featStep: Distance between grid points
    %   imsize: Image size
    % Output:
    %   pointPositions: Nx2 matrix of grid point coordinates
    [ii, jj] = meshgrid(featStep:featStep:(imsize - featStep));
    pointPositions = [ii(:), jj(:)];
end

function [features, labels] = extract_SIFT_imds(imds, imsize, pointPositions)
    % Extract SIFT features from all images in parallel
    % Input:
    %   imds: ImageDatastore
    %   imsize: Target image size
    %   pointPositions: Grid points for feature extraction
    %   features: Pre-allocated feature matrix
    %   labels: Pre-allocated label matrix

    numImages = numel(imds.Files); % Get the number of images in the IMDS
    features = [];
    labels = [];

    h = waitbar(0, 'Processing...');
    for i = 1:numImages
        % Read and preprocess image
        img = imresize(readimage(imds, i), [imsize imsize]);

        if size(img, 3) > 1
            img = rgb2gray(img);
        end

        % Compute the features
        [imfeatures, ~] = extractFeatures(img, pointPositions, 'Method', 'SIFT');

        features = [features; imfeatures];
        labels = [labels; repmat(imds.Labels(i), size(imfeatures, 1), 1) repmat(i, size(imfeatures, 1), 1)];

        waitbar(i/numImages, h)
    end
    close(h)
end

function [BoW, labels] = create_bow(IDX, labels_fe, imds, K)
    % Create Bag of Words representation using cluster assignments
    % Input:
    %   IDX: Cluster assignments from k-means
    %   labels_fe: Feature labels
    %   imds: ImageDatastore
    %   K: Number of clusters
    % Output:
    %   BoW: Bag of Words representation
    %   labels: Image labels

    numImages = numel(imds.Files);
    BoW = zeros(numImages, K);
    labels = zeros(numImages, 1);
    
    parfor i = 1:numImages
        idx = find(labels_fe(:, 1) == imds.Labels(i) & labels_fe(:, 2) == i);
        imfeaturesIDX = IDX(idx);
        H = histcounts(imfeaturesIDX, 1:K+1) ./ numel(imfeaturesIDX);
        BoW(i, :) = H;
        labels(i) = imds.Labels(i);
    end
end