# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [*git fork*](https://github.com/game-of-drones/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the *numpy* library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validating examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32)
* Number of classes = 43


#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

* First, I imported the signnames.csv file, and created a label_to_name `map`. 
* Then, for each label, I randomly chose one image for each label and plotted them with corresponding name.
* Finally, I use a bar chart to plot the number of images for each traffic sign in the training set. As we can see, the distribution of number of samples is not very uniform.

![alt text](./report/one_sign_for_each_label.png)
![alt text](./report/number_of_each_sign.png)

---
### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the `Pre-process the Data Set` Section of the IPython notebook.

[//]: # (As a first step, I decided to convert the images to grayscale because ... Here is an example of a traffic sign image before and after grayscaling. ![alt text][image2])

[//]: # (As a last step, I normalized the image data because ...)

Initially I tried to convert the images to grayscale and train the LeNet. However, the accuracy was not very satisfactory. As a result, I decided to train the net with colored image directly. The result of normalization is that the value of each pixel is between -1 and 1.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

[//]: # (The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.)  

[//]: # (To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...)

Different from the MNIST example in the LeNet lab, there is already validation data in the downloaded image set. 

My final training set had 34799 images. My validation set and test set had 4410 and 12630 images, respectively.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:


| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x3 RGB image                             | 
| Convolution 5x5       | 1x1 stride, VALID padding, outputs 28x28x6    |
| Activation: RELU      |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x6                  |
| Dropout               | Keep Probability 0.74                         |
| Convolution 5x5       | 1x1 stride, VALID padding, outputs 10x10x16.  |
| Activation: RELU      |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Fully connected 1     | output: 120                                   |
| Fully connected 2     | output: 84                                    |
| Fully connected 3     | output: 43 (=n\_class)                        |
| Softmax               | etc.                                          |

Note that I added dropout
#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Training the model is the process of minimizing a cost function. From the following definition, we can see clearly that we are minimizing the mean of cross entropy between the softmax of logits and the one hot of labels.

 > cross\_entropy = tf.nn.softmax\_cross\_entropy\_with\_logits(logits=logits,labels=one\_hot\_y)
 >
 > loss\_operation = tf.reduce\_mean(cross\_entropy)
 >
 > optimizer = tf.train.AdamOptimizer(learning\_rate=rate)
 >
 > training\_operation = optimizer.minimize(loss\_operation)


To train the model, we go through the training set multiple times (defined in EPOCHS). In each epoch, we take a small batch (size defined in BATCH\_SIZE) of input data, run the optimization to update the parameters a little bit (defined by learning rate), and repeat on the next small batch till we finished the data set.

In my model, the hyper parameters I used are:
* BATCH_SIZE = 64
* EPOCHS = 40
* rate = 0.001*0.8
* sigma = 0.05

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.945
* test set accuracy of 0.931

---
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text](./examples/download_32x32/1.jpg)
![alt text](./examples/download_32x32/2.jpg)
![alt text](./examples/download_32x32/3.jpg)
![alt text](./examples/download_32x32/4.jpg)
![alt text](./examples/download_32x32/5.jpg)

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image                   |     Prediction                        | 
|:-----------------------:|:-------------------------------------:| 
|1. RightofWay next inter | RightOfWay next intersection          | 
|2. SpdLmt 50 km/h        | SpdLmt 80 km/h                        |
|3. Caution               | Caution                               |
|4. Road Work             | Road Work                             |
|5. 30 km/h               | SpdLmt 50 km/h                        |

The model was able to correctly guess 3 of the 5 traffic signs (sign 1, 3, and 4), which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The following image shows, for each picture downloaded from internet, the probability of the top 5 predictions.

![alt text](./report/top_5_prediction.png) 
 
* For the first image, the model predicts correctly with very high confidence (almost 100%). 
* For the second image, the model predicts incorrectly, with a relatively high confidence.
* For the third image, the model predicts correctly with a relatively high confidence.
* For the fourth image, the model predicts correctly, but the confidence is relatively low. A wrong prediction has just a little lower probability.
* For the fifth image, the model predicts incorrectly with a very high confidence.