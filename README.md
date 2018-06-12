# **Project 3 - Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./writeup_images/1_nvidia_CNN_model.png "Nvidia CNN model"
[image2]: ./writeup_images/2_final_model.png "Final model achitecture"
[image3]: ./writeup_images/3_example_all_camera_images.png "Example all camera images"
[image4]: ./writeup_images/4_raw_steer_angle_histogram.png "Raw steering angle histogram"
[image5]: ./writeup_images/5_augmented_steer_angle_histogram.png "Augmented steering angle histogram"
[image6]: ./writeup_images/6_sample_flip.png "Sample flipped image"
[image7]: ./writeup_images/7_sample_bright.png "Sample brightness adjusted image"
[image8]: ./writeup_images/8_sample_gaussian_blur.png "Sample Gaussian blurred image"
[image9]: ./writeup_images/9_sample_crop.png "Sample cropped image"
[image10]: ./writeup_images/10_sample_resize.png "Sample resized image"
[image11]: ./writeup_images/11_sample_colorspace.png "Sample color spaced image"
[image12]: ./writeup_images/12_model1_mse.png "Model 1 training versus validation mean squared error"
[image13]: ./writeup_images/13_model2_mse.png "Model 2 training versus validation mean squared error"


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* P3_BehavioralCloning.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model2.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model2.h5
```

#### 3. Submission code is usable and readable

The [`P3_BehavioralCloning.ipynb`](https://github.com/viralzaveri12/CarND-P3-BehavioralCloning/blob/master/P3_BehavioralCloning.ipynb) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First, I started with the basic LeNet architecture and only center images to get an overall idea about the behavior. These selections were not enough to keep the vehicle on track, and realized that model needs more training data, employ some image preprocessing techniques and use more powerful architecture than LeNet.

The project instructions from Udacity suggest starting from a known self-driving car model and provided a link to the [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), as shown below.

![alt text][image1]

Thus, I started with the Nvidia model and included the following layers:
- image normalization using a Keras Lambda function
- three 5x5 convolution layers with 2x2 striding
- two 3x3 convolution layers
- three fully-connected layers
- final layer is the fully-connected layer with a single output neuron

The paper does not mention any sort of activation function or means of mitigating overfitting, so I added the following to the model:
- RELU activation functions after each layer
- maxpooling after each convolutional layer
- dropout (with a keep probability of 0.2) after each layer

The Adam optimizer was chosen with default parameters along with the loss function of mean squared error (MSE).

Below is the mean squared error loss per epoch for model1.

![alt text][image12]

The above model description comprised the `model1.h` as the starting point. This model performed well on the test track but failed in the challenge track as seen in this [video](https://github.com/viralzaveri12/CarND-P3-BehavioralCloning/tree/master/output_videos/model1_challenge.mp4 "Model1 challenge video"). Even though the mean squared error loss is low for both testing and validation, model failing on the challenge track immediately on the first turn suggests model failed to generalize means model is underfitting.

#### 2. Attempts to reduce overfitting in the model

Strategies implemented to avoid underfitting, combat overfitting, and otherwise attempt to get the car to drive more smoothly are:
* Removing dropout layers and adding L2 regularization (lambda of 0.001) to all model layers - convolutional and fully-connected
* Replacing RELU activation functions with ELU activation functions to all model layers - convolutional and fully-connected

#### 3. Model parameter tuning

Adjust learning rate of Adam optimizer to 0.0001 (rather than the default of 0.001).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. For training the model, I used the data provided by Udacity. 

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Starting with Lenet architecture and only the center camera images proved insufficient to keep the vehicle on track. Thus, I tried several other combinations of model and set of camera images as described below:

1. LeNet architecture with center, left, and right camera images (Udacity data)
2. Nvidia architecture with center camera images (Udacity data)
3. Nvidia architecture with center, left, and right camera images (Udacity data)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The 3rd combination of Nvidia architecture with center, left, and right camera images (Udacity data) gave the best results.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

**Step 4** of `P3_BehavioralCloning.ipynb` defines the final model architecture as described in the image below:

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

Udacity provided dataset consisting of center, left, and right camera images and steering angle measurements.

![alt text][image3]

In **Step 0** of `P3_BehavioralCloning.ipynb`, exploring the dataset shows that steering angle measurement of 0.0 is heavily represented when compared to other angle measurements across the range as shown in the histogram below.

![alt text][image4]

Training the NN model with such data would yield a model heavily biased toward driving straight and not knowing much what to do in case of the vehicle swerves off the road. Thus, we would need to augment the dataset labels such that there is enough distribution of data for all the left and right steering angle measurement across the range.

#### Data Augmentation

**a) Using all center, left, and right camera images -** In the `driving_log.csv`, the steering angle values are recorded along with center, left, and right camera images. In the left camera images, the car seems little off to the right of the road, and vice versa for the right camera images, while steering value may still 0.0. Thus, we can augment these images by applying steering angle correction and use the left and right camera images as if they were coming from the center camera.

In **Step 1** of `P3_BehavioralCloning.ipynb`, after augmenting the left and right camera images from the simulator and applying correction to the steering angle measurements, the below histogram shows that there is now enough distribution of data across all the left and right steering angle measurement ranges.

![alt text][image5]

**b) Image flipping -** Furthermore, the first training track is more left turn biased. Hence, in order to increase the generalization of the model, I flipped the images and the respective steering angle measurements.

![alt text][image6]

#### Image Preprocessing

In **Step 2** of `P3_BehavioralCloning.ipynb`, I then applied the following image preprocessing techniques:

**a) Adjust Brightness -** We adjust brightness levels to random number of test images to avoid overfitting and increase generalization of the model.

![alt text][image7]

**b) Apply Gaussian Blur -** We apply Gaussian blurring to random number of test images to avoid overfitting and increase generalization of the model.

![alt text][image8]

**c) Cropping -** The cameras in the simulator capture 160 x 320 pixel images. Not all of these pixels contain useful information, however. The top portion of the image captures trees and hills and sky, and the bottom portion of the image captures the hood of the car. Model might train faster if these pixels are cropped from each image to focus on only the portion of the image that is useful for predicting a steering angle. Thus, cropping 60 pixels from the top and 25 pixels from the bottom, and the resulting image resolution is 75 x 320 pixels.

![alt text][image9]

**d) Resizing -** Resizing the images to 66 x 200 pixel resolution as per the Nvidia model.

![alt text][image10]

**e) Change Color Space -** Changing image color space from RGB to YUV as per the Nvidia model.

![alt text][image11]

#### Training Process

I finally randomly shuffled the dataset and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer with the learning rate of 0.0001.

#### Running the model

Before running the model, I modified the `drive.py` described below:

* Added image preprocessing:

```python
def preprocess(image):
    crop = image[60:135]
    resize = cv2.resize(crop,(200,66))
    new_img = cv2.cvtColor(resize, cv2.COLOR_RGB2YUV)
    return new_img
```
* Adaptive throttle to reduce speed on sharp turns (in case of challenge track):

```python
# Adaptive throttle
        # make sure the model accelerates to a controlled speed and slows down on sharp turns
        if (abs(float(speed)) < 15):
            throttle = 1.
        else:
            if (abs(steering_angle) < 0.1): 
                throttle = 0.3
            elif (abs(steering_angle) < 0.5):
                throttle = -0.1
            else:
                throttle = -0.3
```
The above processes and implementation comprised the `model2.h` architecture. This model performed well both on the test track and the challenge track even after removing dropouts and maxpooling layer. The training and validation data mean squared error loss looks like following:

![alt text][image13]

* Test track using model2  [video](https://github.com/viralzaveri12/CarND-P3-BehavioralCloning/tree/master/output_videos/model2_test.mp4 "Model1 challenge video").
* Challenge track using model2  [video](https://github.com/viralzaveri12/CarND-P3-BehavioralCloning/tree/master/output_videos/model2_challenge.mp4 "Model1 challenge video")
	* In challenge track the vehicle gets stuck at two instances 1) 1:38 - 1:42 and 2) 2:06 - 2:08 where I had to enter the manual mode to set the vehicle back on track. I would say this is acceptable considering the complexity of this track (very sharp turns and high grade) and also the fact that the model did not see this track at all in training.

### Conclusion

This project very much reiterated that it really is all about the data. Making changes to the model rarely seemed to have quite the impact that a change to the fundamental makeup of the training data typically had.

I could easily spend hours upon hours tuning the data and model to perform optimally on both tracks, but to manage my time effectively I chose to conclude my efforts as soon as the model performed satisfactorily on both tracks. I fully plan to revisit this project when time permits.

One way that I would like to improve my implementation is improve the training dataset distribution. As it is currently implemented, I only augmented the left and right turn measurement correction in order to use left and right camera images. Though this augmentation yielded better distribution than just using the center camera images, the distribution was still fairly unbalanced as measurements of -0.25, 0.0, and 0.25 were the only ones heavily represented across the range of -1.0 to 1.0.

I enjoyed this project thoroughly and I'm very pleased with the results. Training the car to drive itself, with relatively little effort and virtually no explicit instruction, was extremely rewarding.