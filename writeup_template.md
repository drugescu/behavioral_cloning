# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./IMG/nn_final.png "Model Visualization"
[image2]: ./IMG/center_cropped_new.jpg "Cropped Training Image"
[image3]: ./IMG/center_cropped_drivepy.jpg "Cropped Driving Image"
[image4]: ./IMG/loss_plots.png "Plot of loss function"
[image5]: ./IMG/center_2nd.jpg "Example center lane driving on second track"
[image6]: ./IMG/center_2016_12_01_13_30_48_287.jpg "Example center lane driving on first track"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model_2nd.py containing the script to create and train the model for the second track (only paths to training data have been changed, same model)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* model_track2.h5 containing a trained convolution neural network  for the second track
* run1.mp4 containing a recording of the first track being driven autonomously by the neural network with no human intervention
* run2.mp4 containing a partial recording of the second track being driven autonomously by the neural network until it falls off the cliff 20% through
* writeup_template.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
python drive.py model_track2.h5
```

#### 3. Submission code is usable and readable

The ```model.py``` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 258, 261, 264). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The model did a fine job with the training data available for track 1.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py lines 269).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

For track 1: used the existing training data with augmentation. Used center, left and right side data (with 0.25 coeffcients) and used all of these flipped as well.

For track 2: recorded and used my own data, only one lap, with several regions of interest recorded a second time (tight turns, cliff jumps). The final track 2 model does a prety good job on the second track except for about 3-4 human interventions required.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with the most basic model and work my way up. 

My first step was to use a convolution neural network model with a single conv layer and a single dense layer similar to what was done in the lessen. This was done to ensure the pieline works and the car attempts to steer iself. It did that, marginally useful. I then replaced this with AlexNet and the car drove the first track wellenough for about 15 seconds. I thought this model might be appropriate because it's small but flexible enough. Further attemts at training did not yield good results so I switched to the nVidia architecture model since this seemed to result in the best autonomous driving experience and it's small enough to run in real time.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80-20%). The initially, after about 10 epochs of training, the training loss was larger than the validation loss, so I added dropout layers to prevent overfitting. This results in training and validation losses about equal after 6-10 epochs of training (depending on... random weights, dog barks, weather and the computer's mood). I've found that after about 4 epochs there was marginal decrease in loss so training time wasn't worth it, which is why the models I have included have only been trained that much.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The second track needs about 3-4 human interventions, but otherwise works well enough.

#### 2. Final Model Architecture

The final model architecture (model.py lines 230-..) consisted of a convolution neural network with the following layers and layer sizes:

* Layer 0: Normalization
* Layer 1: Conv layer with 24 5x5 filters, followed by ELU activation
* Layer 2: Conv layer with 36 5x5 filters, followed by ELU activation
* Layer 3: Conv layer with 48 5x5 filters, followed by ELU activation
* Layer 4: Conv layer with 64 3x3 filters, followed by ELU activation
* Layer 5: Conv layer with 64 3x3 filters, followed by ELU activation
* Layer 6: Flatten to 1164 neurons
* Layer 7: Fully connected layer with 100 neurons, Dropout(0.3) and ELU activation
* Layer 8: Fully connected layer with 50 neurons, Dropout(0.3) and ELU activation
* Layer 9: Fully connected layer with `0 neurons, Dropout(0.3) and ELU activation
* Output: 1 neuron - angle

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I have only captured one lap of driving behaviour on the second track, trying to maintain a speed of less than ```10 mph``` and down to about ```6 mph``` when going uphill. An example of center lane driving on the second track:

![alt text][image5]

To capture good driving behavior, I first recorded a whole lap driving as close to the center as possible, then trained the model and inspected which places caused failure, then re-recorded those. In particular, for the second track, the initial portion of two parallel directions of the track was problematic and the CNN veered left. Several other tight turns proved problematic and the car jumped off the cliff as can be seen in the ```run2.mp4``` recording. I've re-recorded runs on some of these portions and the mode works ok-ish now on the second track, needing 3-4 interventions to stay on track.

To augment the data sat, I also flipped images and angles and used centr, left and right images.

After the collection and augmentation process, I had about 48000 data points for the first track. I then preprocessed this data by resizing and cropping it to nVidia requirements (cropped 60 pixels from the top, 20 from the bottom, resizing at the end the cropped ```320 x 80``` image to ```200 x 66```. As a result, the following 3 images: (one original center lane driving, one cropped training datapoint, one cropped inference datapoint).

![alt text][image6]
![alt text][image2]
![alt text][image3]

```
FROM_TOP = 60
FROM_BOTTOM = 20

new_size_x, new_size_y = 200, 66
old_size_x, old_size_y = 320, 160

icenter = cv2.resize(center_image[0 + FROM_TOP : old_size_y - FROM_BOTTOM, 0 : old_size_x, 0 : 3], (new_size_x, new_size_y), interpolation = cv2.INTER_AREA)
```

The ```drive.py``` file was changed accordingly with this preprocessing step, as outlined in the code:

```
# Preprocess for nVidia pipeline
FROM_TOP = 60
FROM_BOTTOM = 20
image = image.crop((0, FROM_TOP, 320, 160 - FROM_BOTTOM))
image = image.resize((200, 66))
```

One difference between the two images can be seen from the interpolation, present in the model training (since it uses OpenCV) and absent when using inference since PIL was used as a library there.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 6 as evidenced by the diagram I've plotted of training versus validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image4]
