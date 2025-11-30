# bac-model-code

Trained models with initial parameters for classification, object detection and segmentation of breast arterial calcification on FFDM mammograms. Provided as MATLAB live scripts. The accompanying MATLAB code used to train the models is also included.

**Update:** [Our paper](https://doi.org/10.1111/exsy.70069) based on my PhD work was published in Expert Systems in May 2025.

## Pre-processing

Images were pre-processed for the three models which entailed converting the DICOM images into 16-bit pngs and, for classification and object detection, cropping the images to the breast. For both steps we used code made available by [another study](https://github.com/nyukat/breast_cancer_classifier).

For demo purposes, a sample DICOM image with breast arterial calcification, `1-1.dcm`, from the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/nbia-search/?CollectionCriteria=CBIS-DDSM) is included. Its original file location is: \CBIS-DDSM\Calc-Test_P_00646_LEFT_CC\08-29-2017-DDSM-NA-04488\1.000000-full mammogram images-68505.

### DICOM to 16-bit png conversion

Convert DICOM image to 16-bit png using [Pydicom](https://pydicom.github.io/). The bit-depth is set to 16 as that is the bit-depth of the demo image.

````python
import os
import png
import pydicom

def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=16):
    """
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param dicom_filename: path to input dicom file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    image = pydicom.read_file(dicom_filename).pixel_array
    with open(png_filename, 'wb') as myFile:
        writer = png.Writer(
            height=image.shape[0],
            width=image.shape[1],
            bitdepth=bitdepth,
            greyscale=True
        )
    writer.write(myFile, image.tolist())
````
### Crop image

Crop converted image using code from [Wu et al (2019)](https://github.com/nyukat/breast_cancer_classifier):

````python
single_crop.crop_single_mammogram(os.path.join(r, file), # input mammogram_path
                "NO", # horizontal_flip
                view, # view, for the demo image this is 'L-CC'
                file, # cropped_mammogram_path
                'cropped_metadata.pkl', # metadata_path
                100, # num_iterations
                50) # buffer_size
````
| Converted Test Image           | Cropped Test Image            |
| ---------------------- | ---------------------- |
| ![Converted Test Image](1-1.png) | ![Cropped Test Image](croppedImage1-1.png) |

## BAC Segmentation

To try the segmentation network, run the MATLAB live script `DeepLabv3PlusResNet18TrainedNetworkWithInitialParameters.mlx` to create the layer graph, `lgraph`. Assemble the network from the pre-trained layers:
````MATLAB
net = assembleNetwork(lgraph);
````
Convert the cropped image to RGB:
````MATLAB
imagePath = fullfile('croppedImage1-1.png');
grayImage = imread(imagePath);
rgbImage = cat(3, grayImage, grayImage, grayImage);

imwrite(rgbImage, strrep(imagePath,'.png','.tif'),"tif");
````
Read in the cropped tif and apply the segmentation network. 512x512 patches are used:
````MATLAB
im = imread('croppedImage1-1.tif');
figure
imshow(im)

segmentedImage = segmentImagePatchwise(im, net, [512 512]);

seg = labeloverlay(im,segmentedImage);
figure
imshow(seg);

% segmentImagePatchwise performs patchwise semantic segmentation on the input image
% using the provided network.
%
%  OUT = segmentImagePatchwise(IM, NET, PATCHSIZE) returns a semantically 
%  segmented image, segmented using the network NET. The segmentation is
%  performed patches-wise on patches of size PATCHSIZE.
%  PATCHSIZE is 1x2 vector holding [WIDTH HEIGHT] of the patch.

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

````
The network predicts for three classes: BAC, background and breast tissue:
|Test Image           | Prediction            |
| ---------------------- | ---------------------- |
| ![Test Image](testImage.png) | ![Segmented Test Image](segmentedTestImage.png) |

## BAC Object Detection

The dataset used in our study came in four different sizes so all images were cropped as above and padded with zeros using [Pillow](https://pypi.org/project/Pillow/) to a size of 4140x3372. The code below is modified to pad one image, `croppedImage1-1.png`:

````python
from PIL import Image
import os

maxHeight = 4140
maxWidth = 3372
colour = 0

def add_margin(pil_img, top, right, bottom, left, colour):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), colour)
    result.paste(pil_img, (left, top))
    return result

new_path = 'C:\\Users\\user\\Desktop'
files = []
# r=root, d=directories, f = files.
for r, d, f in os.walk(new_path):
    for file in f:
        if file.endswith('croppedImage1-1.png'):
            im = Image.open(os.path.join(r, file))
            width, height = im.size;
            if width % 2 != 0:
                width_padding = maxWidth - width
                left = int((width_padding - 1)/2)
                right = left + 1
            else:
                width_padding = maxWidth - width
                left = int(width_padding/2)
                right = left
            if height % 2 != 0:
                height_padding = maxHeight - height
                top = int((height_padding - 1)/2)
                bottom = top + 1
            else:
                height_padding = maxHeight - height
                top = int(height_padding/2)
                bottom = top
            print('Adding margin...')
            im_new = add_margin(im, top, right, bottom, left, colour)
            print('Added margin...')
            print('Saving image...')
            im_new.save('C:\\Users\\user\\Desktop\\' + str(file).replace('cropped', 'paddedCropped'),
                        quality=100)
            print('Image saved...')
````
Images were reduced to 70% size due to "out of memory" errors on the GPU. A trained yolov4ObjectDetector is provided. The detector performed poorly with detections only occurring at low thresholds, i.e. 0.001:

````MATLAB
% load trainedDetector
load("yolov470epochs.mat");

% read and resize image
I = imread("paddedCroppedImage1-1.png");

I = imresize(I, [2898 2360]);
figure
imshow(I)

% run object detector
[bboxes,scores,labels] = detect(trainedDetector, I, 'Threshold', 0.001);

% view results
I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)
````
|Resized Image           | Prediction            |
| ---------------------- | ---------------------- |
| ![Resized Image](resized70.png) | ![Object Detection Image](objectDetection.png) |

## BAC Classification
To try the classification network, run the MATLAB live script `ResNet22TrainedNetworkWithInitialParametersUpdated.mlx` to create the layer graph, `lgraph`. Assemble the network from the pre-trained layers:
````MATLAB
net = assembleNetwork(lgraph);
````
For the classifier, images were also reduced to 70% size due to "out of memory" errors on the GPU.
````MATLAB
% read and resize image
I = imread("paddedCroppedImage1-1.png");

I = imresize(I, [2898 2360]);
figure
imshow(I)
````
The classification model classifies whole images as 'BAC' or 'NON_BAC':
````MATLAB
>> net.Layers

ans = 

  85×1 Layer array with layers:

     1   'Image_input_1'           Image Input             2898×2360×1 images with 'zerocenter' normalization
    ...
    85   'classoutput'             Classification Output   crossentropyex with classes 'BAC' and 'NON_BAC'
````
Run the model on the demo image and view the results. The model correctly classifies the demo image as BAC+ve:
````MATLAB
[Y, scores] = classify(net, I);

>> Y

Y = 

  categorical

     BAC

>> scores

scores =

  1×2 single row vector

    1.0000    0.0000
````




