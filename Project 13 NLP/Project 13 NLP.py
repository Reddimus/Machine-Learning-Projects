'''
This code performs a sentiment analysis on restaurant reviews. The code reads a dataset of 1000 
restaurant reviews stored in a file named "Restaurant_Reviews.tsv" using pandas and stores it 
in a variable called "dataset". The code then pre-processes the text data in "dataset" by 
removing unwanted words (stopwords), stemming the remaining words, and converting all the words 
to lowercase. The pre-processed data is stored in a list called "corpus".

Next, the code creates a bag of words model from the "corpus" using CountVectorizer from the 
scikit-learn library. The bag of words model is stored in the variable "X". The code also 
separates the dependent variable "y" which is the sentiment of the review (positive or negative).

The code then splits the "X" and "y" data into training and test sets using train_test_split 
from scikit-learn. The training data is used to train a Gaussian Naive Bayes classifier which 
is stored in the variable "classifier". The classifier is then used to predict the sentiment 
of the reviews in the test set and the predictions are stored in "y_pred".

Finally, the code evaluates the classifier's performance by creating a confusion matrix "cm" 
from the true sentiment values "y_test" and the predicted sentiment values "y_pred" using 
confusion_matrix from scikit-learn.
'''

# Natural Language Processing

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
# quoting = 3 is for ignoring "" for our safety.
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t',quoting = 3)
print("Restaurant_Reviews.tsv dataset:")
print(dataset)

# Cleaning the text
# stopwords is a list of unwanted words like the,and,of,etc...
# corpus is a collection of text.
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
# Stemming means taking the root of the word eg. loved, loving, will love -> love
# This will reduce different versions of the same word and will hence reduce the sparsity of matrix
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range (0, 1000):
    # Removing unnecessary punctuations and numbers except letters and replacing removed words with space.
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Converting review to lowercase
    review = review.lower()
    # Converting review to list(of strings)
    review = review.split()
    # Loop through all words and keep those which are not in stopwords list.
    # set is much faster than a list and is considered when the review is very large eg. an article,a book
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Joining back the review list to a string with each word seperated by a space.
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words Model
# Bag of Words Model is a sparse matrix where each row is the review and each column is a unique 
# word from the reviews.
# Tokenization - process of taking all unique words of reviews and creating columns for each word.
# Since this a problem of classification we have dependent and independent variables and each 
# unique word/column is like an independent variable and the review(good/bad) depends on these words.
from sklearn.feature_extraction.text import CountVectorizer
# max_features keeps most frequent words and removes least frequent words (extra cleaning)
# max_feature reduces sparsity, increases precision, better learning and hence better prediction.
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)