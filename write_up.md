#**Traffic Sign Recognition** 

##Writeup Template

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

[image1]: ./examples/barplot.png "Visualization"
[image3]: ./examples/grayscale.jpg "Grayscaling"
[image2]: ./examples/colored.jpg "colored"
[image4]: ./examples/Einbahnstrasse.jpg "One-Way"
[image5]: ./examples/stop.jpg "Stop"
[image6]: ./examples/vorfahrt.jpg "Priority"
[image7]: ./examples/Do-Not-Enter.jpg "Do not enter"
[image8]: ./examples/vorfahrt-kreuzung.jpg "Right-Of-Way"
[image9]: ./examples/prob1.png "prob1"
[image10]: ./examples/prob2.png "prob2"
[image11]: ./examples/prob3.png "prob3"
[image12]: ./examples/prob4.png "prob4"
[image13]: ./examples/prob5.png "prob5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/torbenf86/T1-P2-Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used basic Python commands to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The size of validation set is 4410
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the number of training images for each label. The label number corresponds to a specific traffic sign, which can be found in this csv-file:
[project code](signnames.csv)


![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the provided paper showed that there is no advantage using all three channels. This only increases the input dimension. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]


As a last step, I normalized the image data to [-1,1] by using a normalizing function of OpenCV. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data).

I tried to augment the data set, as the labels are not equally distributed in the training set. There are some sign which can be flipped vertically and horizontally without chaning the meaning (e.g. priority road). However, afterwards it was not really more equally distributed and the achieved accuracy did not improve. So I decided to drop this part, and concentrate on other parts to improve the accuracy.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x32   |
| RELU					|												|
| Local Response Normalization | |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flattening | outputs 800
| Fully connected		| outputs 800      									|
| RELU					|												|
| Fully connected		| outputs 400      									|
| RELU					|												|
| Fully connected		| outputs 43      									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an AdamOptimizer like shown in the units of Udacity. I increased the number of epochs to 100 to see eventual changes. I modified the learning rate very often, and the best outcome in this case was 1e-3.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* training set accuracy of 99%
* validation set accuracy of up to 97% 
* test set accuracy of 94%


* What was the first architecture that was tried and why was it chosen?

I chose LeNet as a base architecture since it was introduced in the course, and seemed to be a good starting point. 

* What were some problems with the initial architecture?

The validation accuracy was stuck at about 90%, while the training accuracy was at 100%. This seemed to indicate an overfitting of the network.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I added two dropout layers after in the fully connected layers and added a normalization to prevent overfitting. Additionally, I changed the input channels of the first two convolutional layers from (6,16) to (16,32). This led to also to larger input of 800 at the first fully connected layer. Therefore I added another fully connected layer. This seemed to improve the network since I obtained at validation accuracy of 97% and test accuracy of 94%.

* Which parameters were tuned? How were they adjusted and why?

I tuned the learning rate, and the initialization of the weights because they have a big influence how the algorithm achieves the minimum of the loss. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it is not part of the training set, but looks similar to another sign in the training set.  

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| One-Way      		| Stop 									| 
| Stop Sign      			| Stop Sign 										|
| Priority Road | Priority Road
| No Entry					| No Entry											|
| Right-Of-Way	      		| Right-Of-Way					 				|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80% (keeping in mind that the model does not know about the first sign).
However, I would have expected that the model detects a sign which might be similar to sign number one, like another sign showing an array. The result with the highest probabilty was the stop sign (see below).

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The following links show bar plot of the predictions for each sign.
Sign 1 (One-Way) : 
![alt text][image9]
Sign 2 (Stop) : 
![alt text][image10]
Sign 3 (Priority) : 
![alt text][image11]
Sign 4 (No Entry) : 
![alt text][image12]
Sign 5 (Right-Of-Way) :
![alt text][image13]


