# Text Emotion Detection - A NLP Project

# Requirements
```python
pip install nltk
pip install textblob
pip install sklearn
pip install pandas
pip install numpy

py -m textblob.download_corpora
```

# How it works
## 1. Import all The Library Required
```python
import pandas as pd # To read file from csv
import numpy as np # To calculate stuffs
import sklearn # Machine learning library
from nltk.corpus import stopwords # To remove stopwords
from textblob import Word # Simple text processing
import re # Regex for Python

from sklearn import preprocessing # Used for label encoder
from sklearn.model_selection import train_test_split # Splitting between training and testing data
from sklearn.feature_extraction.text import CountVectorizer # Vectorize words
from sklearn.metrics import accuracy_score # Calculate the accuract score
from sklearn.linear_model import SGDClassifier # Stochastic Gradient Descent
```
We import all the library that are required to run the program.
## 2. Loading and Removing All Irrelevant Rows
```python
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
```
We remove all the irrelevant data and keep the useful data such as the content and the sentiment for each word.
## 3. Text Processing
```python
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
```
We process the text such removing punctuation etc.
## 4. Text Sorting
```python
# Find the top 10,000 rarest words appearing in the data
freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

# Removing all those rarely appearing words from the data
freq = list(freq.index)
data['content'] = data['content'].apply(
    lambda x: " ".join(x for x in x.split() if x not in freq))
```
The least 10000 used words are removed, since it won't or have little effect to the training.
## 5. Data Labelling and Data Splitting
```python
# Encoding output labels 'sadness' as '1' & 'happiness' as '0'
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)

# Splitting into training and testing data in 90:10 ratio
X_train, X_val, y_train, y_val = train_test_split(
    data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)
```
We encode the sentiment sadness as 1 and happiness as 0, then we split the dataset into 90:10 ratio. 90 For training and 10 for testing.
## 6. Extracting Count Vector
```python
# Extracting Count Vectors Parameters
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])
X_train_count = count_vect.transform(X_train)
X_val_count = count_vect.transform(X_val)
```
Vectorizing the vocabulary
## 7. 


Stop words are natural language words which have very little meaning
Textblob common text processing
Sklearn Preprocessing Label Encoder = Encode target labels with value between 0 and n_classes-1.

Notes:
This codes aren't mine, but I have modified it a bit

Original Sources:
https://github.com/aditya-xq/Text-Emotion-Detection-Using-NLP
https://medium.com/the-research-nest/applied-machine-learning-part-3-3fd405842a18
