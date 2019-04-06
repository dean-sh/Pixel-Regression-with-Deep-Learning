# PixelNet - Pixel prediction with Deep Learning


## **The Challenge**:

This project's goal is to model a deep learning network to estimate the center pixel of an infra-red (IR) image.

<img src="https://github.com/dean-sh/Pixel-Regression-with-Deep-Learning/blob/master/images/train_img1.png" width="500" height="450" />

## The Solution:
The work on this project was divided to several categories.

### **Data Pre-processing**

 1. The dataset contains Infrared gray-scale 31x31 images, with values ranging from ~500 up to ~3000. Those are not RGB values but intensity values for each pixel. 
 2. experimenting with IR images - the train_set includes 10,000 images, while the test_set has 4000.
 3. Augmentation was made by "Sliding"over the 31x31 images and taking small patches to get  11x11, 9x9, 7x7, 5x5 images. 
	 This *expanded the data significantly* to a total of 375625 images from 10,000 (3700% increase!) for 7x7 patch_size. see: `data_augmentation(full_image_array, augment_size)`.
 4. Train-Val-Test Split by using sklearn. see `input_handling_and_saving`.
 5. Min-Max Normalization: I've normalized the images to `[0,1]` values to help the optimization. see `data_normalization(x, min, max)`.
 
### **Network models**
 
I've been experimenting with different network architectures and training schemes to optimize the performance of the model.
 1. **Pixel Classification** - Using classic deep learning classification models, with Softmax activation on a `3000x1` output vector, where the index corresponds to the pixel value.
 2. **Fully Connected Regression** - Using several layers in a Fully-Connected neural network, where the last layer has 1 neuron as a regression problem.
 3. **CNN Regression** - Using several Conv2D layers, some MaxPooling and RelU activation, Flattened and connected to FC layers as in #2. The idea is that the CNN filters will learn spatial information from the real-world images and help the regression accuracy.	

**Parameters Tuning**
 1. Optimizer- **Adam**, SGD, adaGrad. 
 2. Loss: **MSE**, MSLE, MAE as losses for regression.
 3. **Batch Size - 64** was proven to gain the lowest error. 
 4. R2 as Train and Validation Metrics 

Libraries used in the project:

 - Tensorflow
 - Keras
 - numpy
 - sklearn
 - Pandas
 - SciPy
 - matplotlib
 - statistics
 - ngrok localhost tunnel
 
 **Model Architecture:**
 
<img src="https://github.com/dean-sh/Pixel-Regression-with-Deep-Learning/blob/master/Architecture/train_img1.png" width="500" height="450" />

