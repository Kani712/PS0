'''
Name : Kanimozhi Kanagaraj
'''


#Importing libaries

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np

def predict1(vector):
    return np.ones(vector.shape[0])

def load(path):
    with open(path) as f:
        reviews = [line.strip() for line in f]
        labels = np.array([int(line[0]) for line in reviews])
    return reviews, labels

def output_percentage(strategy, dataset, baseline, predictions):
    accuracy = accuracy_score(baseline, predictions) * 100
    print(f"Accuracy of {strategy} ({dataset}): {accuracy:.3f}%") #getiing in % with 3 values after decimal

def main(vectorizer):
    train_reviews, train = load("train.txt")
    val_reviews, test = load("val.txt")
    #print(train_reviews)
    #print(train)
    # create the transform
    #print(vectorizer)
    # create the tokenizer and get the words
    vectorizer.fit(train_reviews)
    
    training = vectorizer.transform(train_reviews)
    testing = vectorizer.transform(val_reviews)

    # The prediction of one always like
    output_percentage("predicting label 1  always like", "training", train, predict1(training))
    output_percentage("predicting label 1  always like", "testing", test, predict1(testing))

    text_classifier = MultinomialNB()
    text_classifier.fit(training, train)

    output_percentage("naive Bayes", "training", train, text_classifier.predict(training))
    output_percentage("naive Bayes", "testing", test, text_classifier.predict(testing))

    text_classifier = LogisticRegression(solver='liblinear')
    text_classifier.fit(training, train)

    output_percentage("logistic regression", "training", train, text_classifier.predict(training))
    output_percentage("logistic regression", "testing", test, text_classifier.predict(testing))

    # create the transform
    #vectorizer = CountVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words='english', ngram_range=(2, 2))
    #for bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    # create the tokenizer and get the words
    vectorizer.fit(train_reviews)

    training = vectorizer.transform(train_reviews)
    testing = vectorizer.transform(val_reviews)

    text_classifier = LogisticRegression()
    text_classifier.fit(training, train)

    output_percentage("logistic regression (binomial)", "training", train, text_classifier.predict(training))
    output_percentage("logistic regression (binomial)", "testing", test, text_classifier.predict(testing))




if __name__ == "__main__":
     vectorizer = CountVectorizer()
     vectorizer_1 = CountVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words='english')
     print('\nWith stopwords : \n')
     with_stop_words = main(vectorizer)
     print('\nWithout stopwords : \n')
     without_stop_words = main(vectorizer_1)
