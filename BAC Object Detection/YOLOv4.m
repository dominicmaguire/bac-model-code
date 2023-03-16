%% Load Data

% get already randomly chosen and augmented images n=2604
trainingData = load("C:\Users\dominic\code\matlab\LiveScripts\Utils\bboxesAllTrainingTable.mat");
trainingDataTbl = trainingData.trainingDataTbl;

% get test image data n=82
testValData = load("C:\Users\dominic\code\matlab\LiveScripts\Utils\bboxesTestTable.mat");
testValDataTbl = testValData.bboxesTestTable;

idx = floor(0.5 * height(testValDataTbl));

testIdx = 1:idx;
testDataTbl = testValDataTbl((testIdx),:);

validationIdx = idx+1 : idx + floor(0.5 * height(testValDataTbl));
validationDataTbl =  testValDataTbl((validationIdx),:);

%% Use imageDatastore and boxLabelDatastore to create datastores for loading 
% the image and label data during training and evaluation.

imdsTrain = imageDatastore(trainingDataTbl{:,"imageFilename"});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,"BAC"));

imdsValidation = imageDatastore(validationDataTbl{:,"imageFilename"});
bldsValidation = boxLabelDatastore(validationDataTbl(:,"BAC"));

imdsTest = imageDatastore(testDataTbl{:,"imageFilename"});
bldsTest = boxLabelDatastore(testDataTbl(:,"BAC"));

%% Combine image and box label datastores.

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

%% validate input data

validateInputData(trainingData);
validateInputData(validationData);
validateInputData(testData);

%% Display one of the training images and box labels.

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

%% reset

reset(trainingData);

%% anchors vs meanIoU

maxNumAnchors = 15;
meanIoU = zeros([maxNumAnchors,1]);
anchorBoxes = cell(maxNumAnchors, 1);
for k = 1:maxNumAnchors
    % Estimate anchors and mean IoU.
    [anchorBoxes{k},meanIoU(k)] = estimateAnchorBoxes(trainingData,k);    
end

figure
plot(1:maxNumAnchors,meanIoU,'-o')
ylabel("Mean IoU")
xlabel("Number of Anchors")
title("Number of Anchors vs. Mean IoU")

%% Create a YOLO v4 Object Detector Network
%Specify the network input size to be used for training.
inputSize = [2898 2360 1];

%Use the estimateAnchorBoxes function to estimate anchor boxes based on the size of objects in the training data. To account for the resizing of the images prior to training, resize the training data for estimating anchor boxes. Use transform to preprocess the training data, then define the number of anchor boxes and estimate the anchor boxes. Resize the training data to the input size of the network by using the preprocessData helper function.
rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 6;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

%% Load Feature Extraction Network

BACNet = load('resnet22withWeightsDropout.mat');
mynet = BACNet.lgraph_2;

inputSize = [2898 2360 1];
basenet = BACNet.lgraph_2;
featureExtractionLayers = ["Relu_44","Relu_63"];

layerName = basenet.Layers(1).Name;
newinputLayer = imageInputLayer(inputSize,'Normalization','none','Name',layerName);

% Remove the fully connected layer in the base network.
basenet = removeLayers(basenet,'classoutput');
basenet = replaceLayer(basenet,layerName,newinputLayer);
% Create a dlnetwork object from the layer
dlnet = dlnetwork(basenet);

classes = {'BAC'};

%% Create the YOLO v4 object detector by using the yolov4ObjectDetector function. 

detector = yolov4ObjectDetector(dlnet,classes,anchorBoxes,DetectionNetworkSource=featureExtractionLayers);

%% Analyse network
analyzeNetwork(detector.Network)

%% Training options

options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=4,...
    L2Regularization=0.0005,...
    MaxEpochs=70,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    ValidationData=validationData);

%% Train network

trainedDetector = trainYOLOv4ObjectDetector(trainingData,detector,options);

%% rest data

reset(testData);

%% iterate test images
inputSize = [2898 2360 1];

disp('Start****');
for i = 1:41
    data = read(testData);
    I = data{1};

    origBoxes = data{2};
    annotatedImage = insertShape(I,'rectangle',origBoxes,'LineWidth',10);
    %annotatedImage = imresize(annotatedImage,inputSize(1:2));

    threshold = 0.9;
    [bboxes,scores,labels] = detect(trainedDetector,I,'MinSize',[33 78],'MaxSize', [1481 1080], 'Threshold',threshold, 'SelectStrongest',true);
    disp(strcat('Image ',string(i),': ',string(scores)));
for x = 1:size(bboxes)
    if scores(x) > 0.9
        annotatedImage = insertObjectAnnotation(annotatedImage,'rectangle',bboxes(x,:),scores(x),'LineWidth',10,'Color','red');
    end
end
figure
imshow(annotatedImage)

end
disp('End****');
%% try one image

I = readimage(imdsTest, 18);

[bboxes,scores,labels] = detect(trainedDetector,I,'MinSize',[33 78],'MaxSize', [1481 1080], 'Threshold',0.9);

I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)

%% Evaluate detector
% threshold = 0.01;
% detectionResults = detect(trainedDetector,testData,'MinibatchSize',4,'MinSize',[33 78],'MaxSize', [1481 1080], 'SelectStrongest',true, 'Threshold',[-1, 1]);
% detectionResults = detect(trainedDetector,testData,'MinibatchSize',4,'SelectStrongest',true, 'Threshold',0.5);
detectionResults = detect(trainedDetector,testData,'MinibatchSize',4);

%% precision etc

[ap,recall,precision] = evaluateDetectionPrecision(detectionResults,testData);
[am, fppi, missRate] = evaluateDetectionMissRate(detectionResults,testData,0.5);

%% plot
figure
loglog(fppi, missRate);
grid on
title(sprintf('log Average Miss Rate = %.3f', am))

%% plot

figure
plot(recall, precision)
xlabel("Recall")
ylabel("Precision")
grid on
title(sprintf("Threshold = %.2f | Average Precision = %.2f",threshold, ap))


%%

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    
    data(ii,1:2) = {I,bboxes};
end

end

%% 
function validateInputData(ds)
% Validates the input images, bounding boxes and labels and displays the 
% paths of invalid samples. 

% Copyright 2021 The MathWorks, Inc.

% Path to images
info = ds.UnderlyingDatastores{1}.Files;

ds = transform(ds, @isValidDetectorData);
data = readall(ds);

validImgs = [data.validImgs];
validBoxes = [data.validBoxes];
validLabels = [data.validLabels];

msg = "";

if(any(~validImgs))
    imPaths = info(~validImgs);
    str = strjoin(imPaths, '\n');
    imErrMsg = sprintf("Input images must be non-empty and have 2 or 3 dimensions. The following images are invalid:\n") + str;
    msg = (imErrMsg + newline + newline);
end

if(any(~validBoxes))
    imPaths = info(~validBoxes);
    str = strjoin(imPaths, '\n');
    boxErrMsg = sprintf("Bounding box data must be M-by-4 matrices of positive integer values. The following images have invalid bounding box data:\n") ...
        + str;
    
    msg = (msg + boxErrMsg + newline + newline);
end

if(any(~validLabels))
    imPaths = info(~validLabels);
    str = strjoin(imPaths, '\n');
    labelErrMsg = sprintf("Labels must be non-empty and categorical. The following images have invalid labels:\n") + str;
    
    msg = (msg + labelErrMsg + newline);
end

if(~isempty(msg))
    error(msg);
end

end

function out = isValidDetectorData(data)
% Checks validity of images, bounding boxes and labels
for i = 1:size(data,1)
    I = data{i,1};
    boxes = data{i,2};
    labels = data{i,3};

    imageSize = size(I);
    mSize = size(boxes, 1);

    out.validImgs(i) = iCheckImages(I);
    out.validBoxes(i) = iCheckBoxes(boxes, imageSize);
    out.validLabels(i) = iCheckLabels(labels, mSize);
end

end

function valid = iCheckImages(I)
% Validates the input images.

valid = true;
if ndims(I) == 2
    nDims = 2;
else
    nDims = 3;
end
% Define image validation parameters.
classes        = {'numeric'};
attrs          = {'nonempty', 'nonsparse', 'nonnan', 'finite', 'ndims', nDims};
try
    validateattributes(I, classes, attrs);
catch
    valid = false;
end
end

function valid = iCheckBoxes(boxes, imageSize)
% Validates the ground-truth bounding boxes to be non-empty and finite.

valid = true;
% Define bounding box validation parameters.
classes = {'numeric'};
attrs   = {'nonempty', 'integer', 'nonnan', 'finite', 'positive', 'nonzero', 'nonsparse', '2d', 'ncols', 4};
try
    validateattributes(boxes, classes, attrs);
    % Validate if bounding box in within image boundary.
    validateattributes(boxes(:,1)+boxes(:,3)-1, classes, {'<=', imageSize(2)});
    validateattributes(boxes(:,2)+boxes(:,4)-1, classes, {'<=', imageSize(1)}); 
catch
    valid = false;
end
end

function valid = iCheckLabels(labels, mSize)
% Validates the labels.

valid = true;
% Define label validation parameters.
classes = {'categorical'};
attrs   = {'nonempty', 'nonsparse', '2d', 'ncols', 1, 'nrows', mSize};
try
    validateattributes(labels, classes, attrs);
catch
    valid = false;
end
end