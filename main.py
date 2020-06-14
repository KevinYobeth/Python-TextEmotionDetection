from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import sklearn
from textblob import Word
import re

import nltk
nltk.download('stopwords')
nltk.download('wordnet')


print('Please wait...')

data = pd.read_csv('text_emotion.csv')

data = data.drop('author', axis=1)

# Dropping rows with other emotion labels
data = data.drop(data[data.sentiment == 'anger'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'empty'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'surprise'].index)
data = data.drop(data[data.sentiment == 'love'].index)
data = data.drop(data[data.sentiment == 'hate'].index)
data = data.drop(data[data.sentiment == 'neutral'].index)
data = data.drop(data[data.sentiment == 'worry'].index)

# Making all letters lowercase
data['content'] = data['content'].apply(
    lambda x: " ".join(x.lower() for x in x.split()))

# Removing Punctuation, Symbols
data['content'] = data['content'].str.replace('[^\w\s]', ' ')

# Removing Stop Words using NLTK
stop = stopwords.words('english')
data['content'] = data['content'].apply(
    lambda x: " ".join(x for x in x.split() if x not in stop))

# Lemmatisation
data['content'] = data['content'].apply(lambda x: " ".join(
    [Word(word).lemmatize() for word in x.split()]))

# Correcting Letter Repetitions


def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


data['content'] = data['content'].apply(
    lambda x: " ".join(de_repeat(x) for x in x.split()))

# Code to find the top 10,000 rarest words appearing in the data
freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

# Removing all those rarely appearing words from the data
freq = list(freq.index)
data['content'] = data['content'].apply(
    lambda x: " ".join(x for x in x.split() if x not in freq))

# Encoding output labels 'sadness' as '1' & 'happiness' as '0'
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)

# Splitting into training and testing data in 90:10 ratio
X_train, X_val, y_train, y_val = train_test_split(
    data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

# Extracting Count Vectors Parameters
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])
X_train_count = count_vect.transform(X_train)
X_val_count = count_vect.transform(X_val)

# Linear SVM
lsvm = SGDClassifier(alpha=0.01, random_state=5, max_iter=100, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)
print('lsvm using count vectors accuracy %s' % accuracy_score(y_pred, y_val))

totalSentence = int(input('Total sentence to input: '))

sentences = []
for i in range(totalSentence):
    sentences.append(input(f'Sentence {i + 1}: '))

# tweets = pd.DataFrame(['I am so happy that I am stressed'])
tweets = pd.DataFrame(sentences)

# Text Preprocessing
tweets[0] = tweets[0].str.replace('[^\w\s]', ' ')
stop = stopwords.words('english')
tweets[0] = tweets[0].apply(lambda x: " ".join(
    x for x in x.split() if x not in stop))
tweets[0] = tweets[0].apply(lambda x: " ".join(
    [Word(word).lemmatize() for word in x.split()]))

tweet_count = count_vect.transform(tweets[0])
tweet_pred = lsvm.predict(tweet_count)

print('\nNo | Mood  | Sentence ')
for i in range(totalSentence):
    if (tweet_pred[i] == 0):
        print(f'{i + 1}  | Happy | ', end='')
    else:
        print(f'{i + 1}  | Sad   | ', end='')
    print(sentences[i])
