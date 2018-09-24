# Infrared Pixel Regression Using Deep Learning

This project's goal is to model a deep learning network to estiamte the center pixel of an infra-red (IR) image.

Similar to Super-Resolution, but for the center pixel only: here the goal is to estimate only the center pixel in the image - which is "0" in the train-set.

The input images are 31x31 gray-scale images, where every pixel value is the intensity, ranging up to ~3000.
The output is the estimated value of the center pixel. 

<p float="left">
  <img src="https://github.com/deansh64/Pixel-Regression-Using-Deep-Learning/blob/master/Images/Train_img1.png" width="400" />
  <img src="https://github.com/deansh64/Pixel-Regression-Using-Deep-Learning/blob/master/Images/Train_img2.png" width="400" /> 
</p>
