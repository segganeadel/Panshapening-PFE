- [1 - Data](#1---data)
  - [1.1 - Preprocessing](#11---preprocessing)
    - [1.1.1 - Data choices](#111---data-choices)
    - [1.1.2 - Data preparation](#112---data-preparation)
      - [1.1.2.1 Original data](#1121-original-data)
      - [1.1.2.2 Downsampling the data](#1122-downsampling-the-data)
      - [1.1.2.3 Making the input block](#1123-making-the-input-block)
- [2 - Metrics](#2---metrics)
- [3 - Model](#3---model)

# 1 - Data
## 1.1 - Preprocessing
### 1.1.1 - Data choices
Since we have already prepared data that has been used to train the list of models in the DLPan Data we just need to understand and explain how the data was prepared. 
If needed, we can also prepare the provided data in the same way as the DLPan.

We will be using the [PanCollection](https://github.com/liangjiandeng/PanCollection) data to train and test the models for uniformity and comparability bewteen the models (Trained under the same data and tested under the same data).

PanCollection proivides already prepared training, validation and test data for these following satelites:
  - WorldView-2 
  - WorldView-3
  - Gaofen-2
  - QuickBird

There are two types of testing data provided:
  - Testing data at **Full Resolution** (FR) which is the original PAN and HRMS images (this is the real way the model should be used) no refrerence (GT) is available for this data.
  - Testing data at **Rownsampled Resolution** (RR) which is the PAN and MS images (this is the way the model is trained) and the HRMS image (this is the GT) is available for this data. (the way this data is generated is explained in the next section).

### 1.1.2 - Data preparation
#### 1.1.2.1 Original data
Initially the data is given in pairs of PAN and MS images these two images provided by the satelites will be referred to as Original PAN and HRMS respectively reffering to the native resolution of the images. 

#### 1.1.2.2 Downsampling the data
In order to train a model we need to provide an input and a refrence for it to train, as refrence we use only the HRMS as the models output only the multispectral data.

As input an we use a degraded version of the HRMS that we will refer to as MS and a downsampled version of the Original PAN image that we refer to as PAN.

The downsampling method for the HRMS and Original PAN is done diffrently:
- for the Original PAN we use an ideal filter this results in the PAN image (in our case being downsampled by a factor of 4).

- for the HRMS we apply an MTF (Modulation Transfer Function) to it, provided that the Nyquist frequency for each channel in data from a certain sensor is known we generate a filter that will be applied to the HRMS image to simulate the degradation that would be caused by the sensor. This results in the MS image. 

![alt text](<assests/Screenshot 2024-05-11 162633.png>)
[fast PyTorch implementation from Z-PNN](https://github.com/matciotola/Z-PNN/blob/master/input_prepocessing.py#L134)

#### 1.1.2.3 Making the input block
The input block is made by concatenating the PAN and MS images along the channel axis, this results in a MS+1 channel image that is used as input for the model but since the MS ans PAN images are of diffrent sizes we need to upsample the MS image to the same size as the PAN image.



# 2 - Metrics
# 3 - Model
