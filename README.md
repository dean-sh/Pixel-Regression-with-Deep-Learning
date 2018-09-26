# Pixel-Regression-with-Deep-Learning
This project's goal is to model a deep learning network to estimate the center pixel of an infra-red (IR) image.

## The Solution:
The work on this project was divided to several categories.

 - **Data Pre-processing**
	 1. experimenting with IR images - the train_set includes 10,000 images, while the test_set has 4000.
	 2. Augmentation was made by "Sliding"over the 31x31 images and taking small patches to get  11x11, 9x9, 7x7, 5x5 images. 
	 This *expanded the data significantly* to a total of 375625 images from 10,000 (3700% increase!) for 7x7 patch_size.
	  *(Code is shared in the jupyter notebook)*.
	 4. Train-Val-Test Split by using sklearn.

