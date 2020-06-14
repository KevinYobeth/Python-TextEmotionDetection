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
import pandas as pd
import numpy as np
import sklearn
from nltk.corpus import stopwords
from textblob import Word
import re

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
```
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

Stop words are natural language words which have very little meaning
Textblob common text processing
Sklearn Preprocessing Label Encoder = Encode target labels with value between 0 and n_classes-1.

Notes:
This codes aren't mine, but I have modified it a bit

Original Sources:
https://github.com/aditya-xq/Text-Emotion-Detection-Using-NLP
https://medium.com/the-research-nest/applied-machine-learning-part-3-3fd405842a18
