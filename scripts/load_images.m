function imds = load_images(csvPath, basePath, type)

    % Read CSV file with file names and labels
    info = readtable(csvPath, "ReadVariableNames", false);
    info.Properties.VariableNames = {'FileName', 'Label'};

    fileNames = info{:, 'FileName'};

    if type == DatasetType.train_20 || type == DatasetType.train_un
        % Read all images
        allImages = dir(fullfile(basePath, '*.jpg'));
        allFileNames = {allImages.name}';
    
        % Select only the path of the train image
        filteredFileNames = intersect(allFileNames, fileNames, 'stable');
        filteredFilePaths = fullfile(basePath, filteredFileNames);

        % Create datastore with only the train image
        imds = imageDatastore(filteredFilePaths);
    else
        imds = imageDatastore(basePath);
    end
    
    if type ~= DatasetType.train_un
        % Add the labels only if not train unlabel
        imds.Labels = info{:, "Label"};
    end
end