%% Apply Clahe

    % Specify the folder where the files live.
    myFolder = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\all_data_augmented_clahe';
    % Check to make sure that folder actually exists.  Warn user if it doesn't.
    if ~isfolder(myFolder)
        errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
        uiwait(warndlg(errorMessage));
        myFolder = uigetdir(); % Ask for a new one.
        if myFolder == 0
            % User clicked Cancel
            return;
        end
    end
    % Get a list of all files in the folder with the desired file name pattern.
    filePattern = fullfile(myFolder, '**/*.png'); % Change to whatever pattern you need.
    theFiles = dir(filePattern);
    for k = 1 : length(theFiles)
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
        fprintf(1, 'Now reading %s\n', string(k));
        % Now do whatever you want with this file name,
        % such as reading it in as an image array with imread()
        image = imread(fullFileName);
        processedImage = applyCLAHE(image);
        imwrite(processedImage, fullFileName);
        fprintf(1, 'Now saving %s\n', string(k));
    end


%% Iterate folds and get data

% delete any remaining files before starting
deleteFiles();


for k = 1:10
    
    foldNum = k;
    fprintf('Starting fold %d:', foldNum);
    
    % Load data
    load('caseSeparatedDataFolds.mat');
    
    % get data from struct array
    testBACArray = foldData{1, foldNum}.testBACArray;
    trainBACArray = foldData{1, foldNum}.trainBACArray;
    valBACArray = foldData{1, foldNum}.valBACArray;
    
    testNON_BACArray = foldData{1, foldNum}.testNON_BACArray;
    trainNON_BACArray = foldData{1, foldNum}.trainNON_BACArray;
    valNON_BACArray = foldData{1, foldNum}.valNON_BACArray;
    
    % set BAC files folder
    bacSource = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\all_data_augmented_clahe\BAC';
    
    % copy and move BAC test files
    dest = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Test\BAC\';
    disp('Moving BAC Test files...');
    for i = 1:length(testBACArray)
        copyfile(strcat(bacSource,'\',testBACArray{i}),strcat(dest,testBACArray{i}));
    end
    disp('Finished moving BAC Test files...');
    
    % copy and move BAC train files
    dest = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Train\BAC\';
    disp('Moving BAC Train files...');
    for i = 1:length(trainBACArray)
        % copy image and augmented images
        fileList = dir(strcat(bacSource,'\\*',trainBACArray{i}));
        for j = 1:length(fileList)
            copyfile(strcat(bacSource,'\',fileList(j).name),strcat(dest,fileList(j).name));
        end
    end
    disp('Finished moving BAC Train files...');
    
    % copy and move BAC validation files
    dest = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Validate\BAC\';
    disp('Moving BAC Validation files...');
    for i = 1:length(valBACArray)
        copyfile(strcat(bacSource,'\',valBACArray{i}),strcat(dest,valBACArray{i}));
    end
    disp('Finished moving BAC Validation files...');
    
    % set non-BAC files folder
    nonBacSource = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\all_data_augmented_clahe\NON_BAC';
    
    % copy and move NON_BAC test files
    dest = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Test\NON_BAC\';
    disp('Moving NON_BAC Test files...');
    for i = 1:length(testNON_BACArray)
        copyfile(strcat(nonBacSource,'\',testNON_BACArray{i}),strcat(dest,testNON_BACArray{i}));
    end
    disp('Finished moving NON_BAC Test files...');
    
    % copy and move NON_BAC train files
    dest = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Train\NON_BAC\';
    disp('Moving NON_BAC Train files...');
    for i = 1:length(trainNON_BACArray)
        % copy image and augmented images
        fileList = dir(strcat(nonBacSource,'\\*',trainNON_BACArray{i}));
        for j = 1:length(fileList)
            copyfile(strcat(nonBacSource,'\',fileList(j).name),strcat(dest,fileList(j).name));
        end
    end
    disp('Finished moving NON_BAC Train files...');
    
    % copy and move NON_BAC validation files
    dest = 'C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Validate\NON_BAC\';
    disp('Moving NON_BAC Validation files...');
    for i = 1:length(valNON_BACArray)
        copyfile(strcat(nonBacSource,'\',valNON_BACArray{i}),strcat(dest,valNON_BACArray{i}));
    end
    disp('Finished moving NON_BAC Validation files...');


    %% Load network with initial weights
    load('resnet22WithWeightsDropout.mat')
    inputSize = [2898 2360 1];

    %% replace image input layer
    imageInLayer = imageInputLayer(inputSize,"Name",'Image_input_1','Normalization','rescale-symmetric');
    lgraph_2 = replaceLayer(lgraph_2,'image_input.1',imageInLayer);
    
    
    %% Get image dataset
    optiTrain = imageDatastore('C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Train',...
        'IncludeSubfolders',true,...
        'FileExtensions','.png',...
        'LabelSource','foldernames');
    optiVal = imageDatastore('C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Validate',...
        'IncludeSubfolders',true,...
        'FileExtensions','.png',...
         'LabelSource','foldernames');
    optiTest = imageDatastore('C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\Test',...
        'IncludeSubfolders',true,...
        'FileExtensions','.png',...
        'LabelSource','foldernames');

    
    % Resize Images
    augTrain = augmentedImageDatastore(inputSize, optiTrain);
    augVal = augmentedImageDatastore(inputSize, optiVal);
    augTest = augmentedImageDatastore(inputSize, optiTest);

    % Set training Options
    options = trainingOptions('adam',...
        'InitialLearnRate',0.0001,...
        'ValidationData',augVal,...
        'MaxEpochs',20,...
        "ValidationFrequency",50,...
        "MiniBatchSize",4,...
        "Shuffle","every-epoch");

    % Perform training
    disp('Training network...');
    BACnet = trainNetwork(augTrain, lgraph_2, options);
    disp('Finished training network...');
    
    % Use the trained network to classify test images
    testpreds = classify(BACnet, augTest, 'MiniBatchSize', 4);
    
    % Evaluate performance metrics
    testAccuracy = nnz(testpreds == optiTest.Labels)/length(testpreds);
    
    confMat = confusionchart(optiTest.Labels,testpreds,"RowSummary","row-normalized");

    cm = confMat.NormalizedValues;

    [precision, recall, f1_score] = evaluate(cm);
    
    fprintf('Test Accuracy: %d', testAccuracy);
    fprintf('Precision: %d', precision);
    fprintf('Recall: %d', recall);
    fprintf('F1 Score: %d', f1_score);
    
    s = struct;
    s.testAccuracy = testAccuracy;
    s.precision = precision;
    s.recall = recall;
    s.f1_score = f1_score;
    save(strcat('data', string(foldNum), '.mat'), 's');
    
    % delete files
    disp('Deleting files...');
    
    deleteFiles();
    disp('Finished deleting files...');
    %
    fprintf('Finishing fold %d:', foldNum);
    
    % clear variables
    disp('Clearing variables...');
    clear foldData;
    clear fileList;
    clear i;
    clear j;
    clear foldNum;
    clear A;
    clear testBACArray;
    clear testNON_BACArray;
    clear trainBACArray;
    clear trainNON_BACArray;
    clear valBACArray;
    clear valNON_BACArray;
    clear BACnet;
    clear inputSize;
    clear imageInLayer;
    clear lgraph_2;
    clear bacSource;
    clear nonBacSource;
    clear dest;
    clear optiTrain;
    clear optiTest;
    clear optiVal;
    clear augTrain;
    clear augTest;
    clear augVal;
    clear options;
    clear testpreds;
    clear testAccuracy;
    clear confMat;
    clear cm;
    clear precision;
    clear recall;
    clear f1_score;
    clear s;
    disp('Finished clearing variables...');
end


%% Function to delete files
function deleteFiles()

% Get a list of all files and folders in this folder.
files = dir('C:\OPTIMAM Dataset\BAC-STUDY\data_augmentation_padded\Cross_Val\**\*.*');
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
% Print folder names to command window.
for k = 1 : length(subFolders)
  delete(sprintf('%s\\*.png',subFolders(k).folder))
end
end

%% return precision recall and F1 score
function [precision, recall, f1_score] = evaluate(cm)
    tp = cm(1,1)/sum(cm(:,1));
    fp = cm(2,1)/sum(cm(:,1));
    fn = cm(1,2)/sum(cm(:,2));
    
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1_score = (2 * precision * recall) / (precision + recall);
end
%% Apply CLAHE
function processedImage = applyCLAHE(image)

    processedImage = adapthisteq(image,'ClipLimit',0.01,'Distribution','uniform','NumTiles',[15,15]);

end


    




