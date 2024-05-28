Malicious attack  detection using machine learning

import glob
import time
import pandas as pd
from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
!pip install db-sqlite3


df = pd.read_csv('https://raw.githubusercontent.com/mageshkumar16/ML/main/SQLiV321.csv')
df.head()


df.shape

df.dropna(inplace=True)

**Visualization**

count=df['Label'].value_counts()
fig = plt.figure(figsize = (6, 3))
plt.bar(count.index,count.values)
plt.xlabel("Queries")
plt.ylabel("Count of Queries")
plt.title("Weightage of Queries")
plt.show
count

**Data Preprocessing**

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()

transformed_posts=pd.DataFrame(posts)
y=df['Label']
y

one_hot_encoded_data = pd.get_dummies(df, columns = ['Label'])
one_hot_encoded_data.drop(['Sentence'],inplace=True,axis=1)
y=one_hot_encoded_data.values

y

**Training & Testing**

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformed_posts, y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
#from keras.wrappers.scikit_learn import KerasClassifier

input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(20, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(10,  activation='tanh'))
model.add(layers.Dense(1024, activation='relu'))

model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

classifier_nn = model.fit(X_train,y_train,
                    epochs=1,
                    verbose=True,
                    validation_split=0.2,
                    batch_size=15)

**Prediction**

y_pred=model.predict(X_test)

for i,row in enumerate(y_pred):
   max_element=max(row)
   for j in range(len(row)):
     if row[j]==max_element:
       row[j]=1
     else:
       row[j]=0


y_test=y_test.astype('float32')

y_test

**Classification Report**

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

**HEAT MAP**

from sklearn import metrics
import seaborn as sns
cm=metrics.confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
cm_df = pd.DataFrame(cm,
                     index = ['NQ','SQL','CSS'],
                     columns = ['NQ','SQL','CSS'])
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True)
plt.title('HEATMAP')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

**OUTPUT*

while True:
  URL = "https://basvtech.000webhostapp.com/GetQuery.php"
  r = requests.get(URL)
  soup = BeautifulSoup(r.content, 'html5lib')
  Qry = soup.find(id="Query")
  qy=[]
  qy.append(Qry.text)
  if Qry.text!="0":
      data=pd.DataFrame(qy,columns=['Sent'])
      vectorizee = CountVectorizer()
      poste = vectorizee.fit_transform(data['Sent'].values.astype('U')).toarray()
      print(poste)
      z_pred=model.predict(poste)
      for i,row in enumerate(z_pred):
        max_element=max(row)
        for j in range(len(row)):
          if row[j]==max_element:
            row[j]=1
          else:
            row[j]=0
      max_element=max(max(z_pred))
      for i,row in enumerate(max(z_pred)):
          if row==max_element:
            if i==0:
              Q_type="Normal Query"
            elif i==1:
              Q_type="SQL Injection"
            elif i==2:
              Q_type="XSS Attack"
      URL = "https://basvtech.000webhostapp.com/InsertLog.php?qry="+Qry.text+"&type="+Q_type
      r = requests.get(URL)
      soup = BeautifulSoup(r.content, 'html5lib')
      URL = "https://basvtech.000webhostapp.com/TruncateDB.php"
      r = requests.get(URL)
      soup = BeautifulSoup(r.content, 'html5lib')
#m
model = Sequential()
model.add(layers.Dense(10,  activation='tanh'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
