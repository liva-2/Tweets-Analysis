# Tweets-Analysis
This is a sentiment analysis for more than 4600 tweets to classify the tweets into hate, abusive, and normal.

## Imports:
First I imported all the libraries that I'm going to use:
```python
#data load and preprocessing
import numpy as np
import pandas as pd 
import re 
import string

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import nltk
from nltk.probability import FreqDist
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#ML Prepration
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score 

# Classification Model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

```
## Read and preprocess the data:
* Let's begin with reading the data:
```python
# reading the file
df = pd.read_csv("train.csv")
df.head()
```
* Then I tried to print a sample of the data:
```python
df.Tweet[50]
```
* Now it's time to use the NLP libraries to preprocess the data, first I took off the stopwords, then I used ISRIStemmer to reduce the word to its stem:
```python
# define punctuations
punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' 

# Expande Arabic StopWords
arabic_stopwords = set(nltk.corpus.stopwords.words("arabic"))
data = []
with open('arabic_stop_words.txt',encoding="utf8") as myfile:
    for line in myfile:
        data.extend(map(str, line.rstrip('\n').split(',')))
arabic_stopwords.update(data)

#difine Arabic text stemmer
st = ISRIStemmer()

# 
def clean_text(text):
    text = text.translate(str.maketrans('','', punctuations))
    text = ' '.join([word for word in text.split() if word not in arabic_stopwords])
    text = st.stem(text)
    return text
df['Tweet'] = df['Tweet'].apply(clean_text)
```
* After dealing with the string, now it's time to vectorize each tweet:
```python
#Create 
max_features = 5000
count_vector = CountVectorizer(max_features = max_features)  
PoemBOW = count_vector.fit_transform(df['Tweet']).toarray() 
PoemBOW
# BOW frequency
print(count_vector.vocabulary_)
```
* Since we vectorize the tweet, then we have to encode the classes to have zero string data:
```python
# encode Label Column
encoder = LabelEncoder()
Label_encoder = encoder.fit_transform(df['Class'])
Label_encoder
```
## Train the data:
* The first step of the training is splitting the data into train and test:
```python
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(PoemBOW,Label_encoder, test_size =0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```
* After that I used Logistic Regression, MultinomialNB, Random Forest Classifier, LinearSVC, and GaussianNB algorithms but the highest accuracy was the Logistic Regression with 0.79 accuracy:
```python
logModel = LogisticRegression()
# train model
logModel.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = logModel.predict(X_test) 

# print the accuracy and classification_report
print('Test model accuracy: ',accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
## Test the model
Finally, it's time to test the model and see the results:
```python
df2 = pd.read_csv('train.csv')

#qoute
test_text = [df2['Tweet'][55]]
# حزن
# encoding
test_vector = count_vector.transform(test_text)
test_vector = test_vector.toarray()

## Perform and Evaluate the Model 
text_predict_class = encoder.inverse_transform(logModel.predict(test_vector))
print(test_text,'\n',text_predict_class)
```
