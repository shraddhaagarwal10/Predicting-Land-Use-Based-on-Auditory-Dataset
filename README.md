# Predicting-Land-Use-Based-on-Auditory-Dataset
## Motivation
‘Land use’ is a term used to describe economic and cultural activities (e.g., agriculture, industrial, residential, mining, commercial etc) that are practiced at a given place.
Land use changes occur constantly and can have effects on quality of human and wildlife habitat, air and water quality etc.  
We will start with only 3 regions of classification i.e Industrial, Residential and Commercial..
Since the number of classes are less we are motivated to search for lower complexity and lesser resource-consuming solutions (both memory-wise and computationally)
## Research Gap
No prior dataset for this particular classification.
Large variation in sounds and noises across different regions in India for all three classes.
No clear distinction among Residential, Industrial and Commercial areas.
Sometimes the sound pattern is not identified due to common audio samples among all three classes
## Objective
Finding optimal solution for the classification of the geographical locations based on audio inputs using Artificial Intelligence.
Proposing a deep learning framework with a less resource intensive CNN based model for Acoustic Scene Classification (ASC) that can perform well on low power and real life systems (our model will be device and operating system independent). 
To report accuracy on test subjects which consists of audio samples from cities or particular regions of India. 
We had made sure that our model does not hold bias towards any particular region of India.
## Methodology
### Dataset preparation:
Prepared a brand new benchmark dataset from YouTube and other internet videos. 
Converted the audios (which were taken from YouTube) into mp3 and then into wav format.
Split them into the clips of length 10 seconds (standard chosen for Gammatone Spectrogram).
Cleaned the dataset by considering only those audios which have audible sound.
Collected audio samples from Bhopal for testing data (Bairagarh for Commercial area, Govindpura for Industrial area and IISER  Bhopal Campus for Residential area).
### Pre-Processing
Split multiple channel audio into single channel waveforms.
Converted each waveform into Gammatone spectrograms - visual representation of frequencies of given signal with time.
Images produced are separated into 3 parts:- train, test, validation.
No of samples in training set:-  10,595
     No of samples in validation set:- 1500
     No of samples in test set:- 810 
### Model Selection
We proposed CNN(Convolutional Neural Network) model for building our classifier.
CNN requires images to be fed as input data to perform convolution. To facilitate this, audio needs to be converted to spectrogram images.
CNN is considered to be more powerful than RNN as it includes more feature compatibility when compared to RNN. Furthermore, time series consideration in RNN wasn’t necessarily required as observed from the results of our model.
## Results
Accuracy acquired:-  

98% (if audio acquired from same device)

Despite of feeding images of low resolution (64 x 64)
model was able to provide sparkling results in training and validation 
sets.
This proves Gammatone spectrogram are reliable inputs for any kind 
of Acoustic Scene Classification (ASC) task.

However, we observed that collecting audio from multiple devices
is not recommended if CNN is used, as sound quality of each device
varies and so is the output of gammatone spectrogram.

