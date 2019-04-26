
#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing Dataset
dataset = pd.read_csv('googleplaystore_user_reviews.csv')
dataset = dataset[['Translated_Review','Sentiment']].dropna()
mask = dataset['Sentiment']!= 'Neutral'
dataset = dataset[mask]
dataset['Sentiment'] = dataset['Sentiment'].map({'Positive': 1, 'Negative': 0})
dataset.index=range(32269) #32269
#dataset['Sentiment'] = dataset['Sentiment'].astype('int')

#Cleaning
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
#cc=len(dataset['Translated_Review'])
for i in range (0,32269):
    review = re.sub('[^a-zA-Z]'," ",dataset['Translated_Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    
#Creating Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 47000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[ : ,1].values

#Splitting the dataset into train, test, split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting the Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting the test set result
y_pred = classifier.predict(X_test)

#Viewing the results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


















#(1439+1324)/6453.8
#(718+646)/3226.9
#(1439+1330)/6453.8
#(1488+1065)/6453.8
#(1439+1324)/6453.8














