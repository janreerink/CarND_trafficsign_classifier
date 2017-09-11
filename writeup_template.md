# **Traffic Sign Recognition** 

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

[image0]: ./examples/vis0.png "Example traffic sign"
[image1]: ./examples/vis1.png "Samples per class"
[image2]: ./examples/vis2.png "train-test-validation sizes"
[image3]: ./examples/vis4.png "Augmented image"
[image4]: ./examples/vis5.png "Augmented dataset"

[image5]: ./new_imgs/120.jpg "Augmented dataset"
[image6]: ./new_imgs/animalcrossing.jpg "Augmented dataset"
[image7]: ./new_imgs/construction.jpg "Augmented dataset"
[image8]: ./new_imgs/rightofwayjpg.jpg "Augmented dataset"
[image9]: ./new_imgs/stop.jpg "Augmented dataset"

[image10]: ./examples/features.png "Feature maps"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/janreerink/trafficsign_classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
A sample pictures and its bounding box is shown along with the label belonging to this traffic sign.
![alt text][image0]

The next picture is a bar chart showing the number of samples per class for the whole dataset. Apparently the dataset is imbalanced with some (common) signs having more samples than others.
![alt text][image1]

The next picture shows a bar chart with the proportion of samples in the train, test and validation sets for each class. The split ratios are generally similar, but there are small differences with some classes having fewer test
and validation samples. It looks like the set sizes are multiples of 30.
![Train-test-val splits][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The pictures were converted to grayscale as suggested in the project instructions. The images are normalized as shown in class to have zero mean and equal standard deviation to improve performance of gradient descent.
Later histogram equalization was added as it improves contrasts and leads to slightly better results for new images.

Aside from pre-processing the dataset was also augmented to improve model performance. For every image in the dataset at least one transformed copy was added (to prevent systematically changing some of the classes). 
Classes with few labels had additional images added until a mininum number of samples for each class was obtained. The augmentation consisted of randomly rotating, stretching, zooming in or out and transposing
the original image. This increased model performance both for validation with the original dataset and especially for the new images.


Here is an example of an augmented image:
![alt text][image3]

And here the balanced dataset after augmentation:
![alt text][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers, returning logits that were used for evaluation and
later for calculating probabilities using softmax:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten    			| Outputs 400x1									|
| Fully connected		| Outputs 120x1									|
| RELU					|												|
| Dropout				| Drops keep_prob percent of neurons            |
| Fully connected		| Outputs 84x1									|
| RELU					|												|
| Dropout				| Drops keep_prob percent of neurons            |
| Fully connected		| Outputs 43x1									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I performed a basic grid-search on a range for some hyperparameters (learning rate, drop out probability, batch size) for 50 epochs each. The average validation accuracy of the last 5 epochs was used
to select a combination of parameters. A combination of batch size 128, keep probability of 50% and learning rate of 0.0009 worked well, yielding a validation accuracy of 98% and a test accuracy of 95%. I experimented with changed to the number of conv layers and optimizers, but spending time on preprocessing
and data augmentation seemed to be more effective. Hence I used the Adam optimizer and same number of layers as shown in class. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.5%
* validation set accuracy of 98.1%
* test set accuracy of 95.3%

As shown in the notebook, the test accuracy for some classes (e.g. pedestrians crossing) was considerably worse. Chances are that when few samples are available even image augmentation will not result in perfect classification. 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    The model shown in class seemed suitable for classification of images.
* What were some problems with the initial architecture?
    The architecture initially did not support drop out, which was later added to increase robustness.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    The architecture tended to underfit, mostly hitting a ceiling at around 95% in validation accuracy. Changes to number of layers did not seem to impact performance.
* Which parameters were tuned? How were they adjusted and why?
    Learning rate, batch size and drop out probability were tuned as described above. The model results changed quite a bit with the learning rate and to a lesser degree with batch size. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    The convolution layers worked as filters, identifying geometries in subregions of the image that could be combined into more complex shapes, learned by the model. Dropout layers reduced the tendency of the model to overfit.

If a well known architecture was chosen:
* What architecture was chosen?
    Readily available after introduction in class; performs well on similar issues.
* Why did you believe it would be relevant to the traffic sign application?
    The LeNet architecture was one of the first neural net architectures to achieve the performance levels of humans in image classification.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
    According to literature human performance for the dataset is about 98%, hence a 95% test accuracy of the model is getting close to what we would want to use in self-driving cars. While there is some indication of overfitting 
    (test accuracy lower than validation accuracy) the similar training and validation accuracy demonstrates a certain robustness of the model. This could probably be improved with larger test and validation sets.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The images differ slightly in lighting condition, background, means of sign fixation to the post (i.e. noise), perspective and pixel density (of the original).
Speed limit sign: there is no particular indication why this would be difficult to classify; it is zoomed in more than the training images and somewhat darker.
Animals corssing sign: this image also looks like it might be easy to classify, there only appears to be some noise in the red area of the triangle
Construction work: this image is slightly angled; the sign is fastened to the post at the top and bottom, adding quite a bit of noise
Right of way: this image is taken from up close, leading to a perspective different from those in the training set. However, it has a distinctive shape and few details.
Stop: this image should be easy to classify


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 120  		| Speed limit 20								| 
| Animals    			| Animals                                       |
| Construction 			| Construction                                  |
| Right of Way  		| Right of Way									|
| Stop  	      		| Stop      					 				|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Given the test set accuracy of 95% this does not appear to be a great result. It is possible that the preprocessing
methods used lead to images that are systematically different from the training set. The 120km/h sign looks like it would be easy to predict, it is likely that when compressing the high resolution image to the 32x32
size a distinction between details contained in the sign are harder relative to a classificaiton of larger shapes. The predicted class (20 km/h) speed limit is likely to activate many of the same features that would
be relevant for a 120 km/h sign, suprisingly visualization of the softmax probabilities shows that the classifier is certain. 



####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model seems very certain for each prediction, even those that are wrong. Only in the case of the incorrectly classified 120km/h speed limit we can see that aside from the incorrect prediction (20km/h speed limit
with 93%) there is a 6% probability for the correct label. It looks like the model prefer sparse results even though no explicit steps were (knowingly) taken to induce sparsity. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][image10]

The jupyter notebook shows the filters for the 120 km/h sign. The feature maps seem show the circular shape on the traffic sign as well as the numbers contained within. Some of the maps appear to be more or less random though.


