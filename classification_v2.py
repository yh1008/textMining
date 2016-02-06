__author__      = "Ye Hua (Emily)"

import time 
import pickle
import sys
import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import nltk
import math
import string
from nltk import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#csv file needs latin-1 encoding 
reload(sys)  
sys.setdefaultencoding('latin-1')
stopwords = stopwords.words("english")
tokenizer = RegexpTokenizer(r'\w+')

print 'Loading Data'
## Load data: be careful: first ("header") row is the feature names!  
f = open('IMBD_train.csv')

def pre_process (raw_file):
	vectorizer = CountVectorizer(min_df=1)
	f = raw_file
	sentiment_score = [] 
	# ( eg. ) review_list = ['Review, Sentiment', 'She likes you', 'He hates you']
	review_list = f.read().splitlines() 
	num_reviews = len(review_list)-1
	review_list.pop(0)
	corpus = [] #store list with each item the review
	for i  in range(len(review_list)):
		# ( eg.) review_list[i] = 'she likes you' == each review 
		review = review_list[i].rsplit(',', 1)[0]
		review_tokenized= tokenizer.tokenize(review)
		filtered_review = [word for word in review_tokenized if word not in stopwords]		
		filtered_review = [PorterStemmer().stem_word(word) for word in filtered_review]	
		string = ' '.join(filtered_review)	
		corpus.append(string) #['you like me', 'she hate you']		
		sentiment = review_list[i].rsplit(',',1)[1]	
		sentiment = int(sentiment)
		#review = re.sub(r',\d', "", review_list[i])
		sentiment_score.append(sentiment)
	vectorizer = TfidfVectorizer(min_df=1)
	tfidf = vectorizer.fit_transform(corpus)	
	return tfidf, num_reviews, sentiment_score

def cross_validation(input_train_reviews, input_train_sentiment):
	train_reviews = input_train_reviews
	train_sentiment = input_train_sentiment
	review_train, review_test, sentiment_train, sentiment_test = train_test_split(train_reviews, train_sentiment, test_size = 0.33, random_state = 22)
	return review_train, review_test, sentiment_train, sentiment_test

#random forest/multinomial naive bayes/KNN/LogisticRegression/NuSVM/SVM
def train_model(model_name, bag_of_word_feature, sentiment_score):	
	model_name = model_name
	if model_name == "randon_forest":
		clf = RandomForestClassifier(n_estimators = 100)
	elif model_name == "multinomial_naive_bayes":
		clf = MultinomialNB()
	elif model_name == "KNN":
		clf = KNeighborsClassifier(n_neighbors=5)
	elif model_name == 'logisticRegression':
		clf = LogisticRegression(C=1e5)
	elif model_name == 'NuSVM':
		clf = svm.NuSVC(probability=True)
	elif model_name == 'SVM':
		clf = svm.SVC(probability=True)
	model = clf.fit(bag_of_word_feature, sentiment_score)
	return model

def test_model(model, bag_of_word_feature_test):
	model = model
	bag_of_word_feature_test = bag_of_word_feature_test
	prediction = model.predict(bag_of_word_feature_test)
	return prediction

def plot_auc(fpr,tpr,roc_area):
	fpr = fpr
	tpr = tpr
	roc_area = roc_area
	plt.figure()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_area)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.show()

def show_most_informative_features(vectorizer, clf, n=20):
	vectorizer = vectorizer
	clf = clf
	feature_names = vectorizer.get_feature_names()
	coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
	top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
	for (coef_1, fn_1), (coef_2, fn_2) in top:
		print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def main():
	tfidf, num_reviews, sentiment_score = pre_process(f)
	review_train, review_test, sentiment_train, sentiment_test = cross_validation(tfidf, sentiment_score)
	model = train_model("NuSVM", review_train, sentiment_train)
	prediction = test_model (model, review_test)
	print "prediction: "
	print prediction 
	accuracy = model.score(review_test, sentiment_test)
	print "prediction metrics: "
	print 'accuracy: %0.2f'%accuracy
	# Determine the false positive and true positive rates
	prob = model.predict_proba(review_test)
	fpr, tpr, _ = roc_curve(sentiment_test, prob[:,1])
	#calculate the AUC
	roc_area = auc(fpr, tpr)
	print 'ROC area: %0.2f' % roc_area
	plot_auc(fpr,tpr,roc_area)

if __name__ == '__main__':
	time_start = time.clock()
	print "Computing..."
	main()
	time_elapsed = (time.clock() - time_start)
	print "used: %0.2f seconds"%time_elapsed



