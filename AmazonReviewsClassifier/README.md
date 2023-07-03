# Amazon Reviews Classifier
One of the areas of sentiment analysis application is to understand the correlation of the review text with the review rating given by a user on an online portal such as amazon.com.

This code presents 4 recurrent neural network models (RNN, LSTM, GRU, BiLSTM) to classify the ratings on Amazon reviews by going through the text. All models are written using Pytorch, Torchtext.

# Dataset 
The dataset consists of customer’s reviews and ratings, which we got from Consumer Reviews of Amazon products. The task is to classify the reviews into 5 classes (with 1 being the lowest and 5 being the highest rating a product can get in a review), where ratings constitute the ground truth class labels. 

# Running
Unzip the dataset from the dataset folder then from the src folder run the below command:
```
python AmazonReviewClassifier.py
```
To change the class weights for the training of the models, change the weights param on line 49 to assign weights for each class from 1 through 5.
