%% load data
dataDir = fullfile('F:\');

% set up training and validation datastores
imdir = fullfile(dataDir, "SAMPLE_RGB_200");
imds = imageDatastore(imdir);

valDir = fullfile(dataDir, "SAMPLE_VAL_RGB_200");
imdsVal = imageDatastore(valDir);

classNames = ["background", "BAC", "breast"];
labelIDs = [1 2 0];

% set up training and validation pixel datastores
labelDir = fullfile(dataDir,'SAMPLE_RGB_200_PIXEL_DS');
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

labelDirVal = fullfile(dataDir,'SAMPLE_VAL_RGB_200_PIXEL_DS');
pxdsVal = pixelLabelDatastore(labelDirVal,classNames,labelIDs);

tbl = countEachLabel(pxds);

numberPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / numberPixels;
classWeights = 1 ./ frequency;

% Create training and validation random patch datastores
pixeldsTrain = randomPatchExtractionDatastore(imds,pxds,512, ...
     'PatchesPerImage',128);

pixeldsVal = randomPatchExtractionDatastore(imdsVal,pxdsVal,512, ...
     'PatchesPerImage',128);

%% set up basic deepLabv3+
imageSize = [512 512 3];

numClasses = 3;

lgraph = deeplabv3plusLayers(imageSize,numClasses,'resnet18');

pixelClassLayer = pixelClassificationLayer("Classes",["background" "BAC" "breast"],"ClassWeights",classWeights, "Name", "Seg-Layer");

imageInLayer = imageInputLayer(imageSize,"Name",'Image_input_1','Normalization','rescale-symmetric');
lgraph = replaceLayer(lgraph,'data',imageInLayer);
lgraph = replaceLayer(lgraph,'classification', pixelClassLayer);

%% Train the network.
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',70, ...
    'VerboseFrequency',10,...
    'MiniBatchSize',16, ...
    'Plots','training-progress', ...
    'ValidationData',pixeldsVal);


net = trainNetwork(pixeldsTrain,lgraph,options);

%% test network
classNames = ["background", "BAC", "breast"];
labelIDs = [1 2 0];

imdirTest = fullfile(dataDir, "SAMPLE_TST_RGB_200");
imdsTest = imageDatastore(imdirTest);

testLabelDir = fullfile(dataDir,'SAMPLE_TEST_RGB_200_PIXEL_DS');
pxdsTruth = pixelLabelDatastore(testLabelDir,classNames,labelIDs);


%%
dest = 'F:\PREDICTIONS\';

for i = 1:length(imdsTest.Files)

[im, fileinfo] = readimage(imdsTest,i);

segmentedImage = segmentImagePatchwise(im, net, [512 512]);

fileDesc = strrep(fileinfo.Filename,'F:\SAMPLE_TEST_RGB_200\image','');
imwrite(segmentedImage, strcat(dest,'pred_Label_',fileDesc));

end

%% evaluate segmentation
dest = 'F:\Masks\';

classNames = ["background", "BAC", "breast"];
labelIDs = [1 2 3];

pxdsPred = pixelLabelDatastore(dest,classNames,labelIDs,"FileExtensions",'.tif');
ssm = evaluateSemanticSegmentation(pxdsPred,pxdsTruth);

%% confusion

classNames = ["background" "BAC" "breast"];
normConfMatData = ssm.NormalizedConfusionMatrix.Variables;
figure
h = heatmap(classNames, classNames, 100 * normConfMatData);
h.XLabel = 'Predicted Class';
h.YLabel = 'True Class';
h.Title  = 'Normalized Confusion Matrix (%)';

%%
% segmentImagePatchwise performs patchwise semantic segmentation on the input image
% using the provided network.
%
%  OUT = segmentImagePatchwise(IM, NET, PATCHSIZE) returns a semantically 
%  segmented image, segmented using the network NET. The segmentation is
%  performed patches-wise on patches of size PATCHSIZE.
%  PATCHSIZE is 1x2 vector holding [WIDTH HEIGHT] of the patch.

%   Copyright 1984-2018 The MathWorks, Inc.

function out = segmentImagePatchwise(im, net, patchSize)

[height, width, nChannel] = size(im);
patch = zeros([patchSize, nChannel], 'like', im);

% pad image to have dimensions as multiples of patchSize
padSize(1) = patchSize(1) - mod(height, patchSize(1));
padSize(2) = patchSize(2) - mod(width, patchSize(2));

im_pad = padarray (im, padSize, 0, 'post');
[height_pad, width_pad, ~] = size(im_pad);

out = zeros([size(im_pad,1), size(im_pad,2)], 'uint8');

for i = 1:patchSize(1):height_pad
    for j =1:patchSize(2):width_pad
        patch = im_pad(i:i+patchSize(1)-1, j:j+patchSize(2)-1, :);
        patch_seg = semanticseg(patch, net, 'outputtype', 'uint8');
        out(i:i+patchSize(1)-1, j:j+patchSize(2)-1) = patch_seg;
    end
end

% Remove the padding
out = out(1:height, 1:width);

end