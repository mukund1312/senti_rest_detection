# Bags of words tech
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
import nltk  # this library is used to filter the words that would be a hinderence while training the model to give feedback such as "YES" or "NO" or "POSITIVE" or "NEGATIVE" with word in english
import re  # library used to simplyfy the data set
from nltk.corpus import stopwords
from nltk import corpus
from nltk.corpus.reader import reviews
from nltk.util import pr
import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt

Data_set = pd.read_csv("senti anali\Restaurant_Reviews.tsv",
                       delimiter='\t', quoting=3)  # here we are importing the dataset

# THIS IS CLEANING THE DATA

corpus = []  # this is the list of all words wchich are cleaned
for i in range(0, 1000):  # 1000 is the lenght os the data set and this is the loop for putting all the words from the data set inot the corpus list
    # this is to remove all the puntuations and keeping only the words
    review = re.sub('[^a-zA-Z]', ' ', Data_set['Review'][i])
    review = review.lower()  # converting all the words into lower case to make it more easy
    review = review.split()  # this is to seperate each word individual for stemming
    ps = PorterStemmer()  # this is used for stemming the words example the word loved and love mean the same when the expected output is "POSITIVE" or "NEGATIVE" thus why increase the dimension of the matrix
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    # this line run a for loop for all the words in the dataset which are not a part of the stopwords and then applying the stemming to it and then storing it in the list named reviews
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)


# CREATING THE BAG OF WORDS
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = Data_set.iloc[:, -1].values


# splitting up into train set and the train set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# training the naive bayes model on the train data set
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# predicting the test set
y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# making the confusion matircx
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("[correct prediction of negative veiws,incorrect prediction of positve reviews]")
print("[incorrect predictions of negative reviews ,incorrect predictions os positive reviews]")
print(accuracy_score(y_test, y_pred))

# making a single prediction
new_review = input("enter the review: ")
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word)
              for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
new_y_pred
if(new_y_pred == 1):
    print("POSITIVE REVIEW")

else:
    print("NEGATIVE REVIEW")
