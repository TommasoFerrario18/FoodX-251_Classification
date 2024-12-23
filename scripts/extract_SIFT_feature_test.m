function [features, labels] = extract_SIFT_feature_test(featStep, imsize, imds, K, C)
    % Construction of the grid on which SIFT descriptors will be extracted
    disp('Grid construction')
    pointPositions = generate_grid(featStep, imsize);
    
    % Estrazione features sul training
    disp('estrazione features')
    [features, labels] = extract_SIFT_imds(imds, imsize, pointPositions, K, C);
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

function [features, labels] = extract_SIFT_imds(imds, imsize, pointPositions, K, C)
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

        D = pdist2(imfeatures, C);
        [~, words] = min(D, [], 2);

        H = histcounts(words, 1:K+1) ./ numel(words);

        features = [features; H];
        labels = [labels; imds.Labels(i)];

        waitbar(i/numImages, h)
    end
    close(h)
end