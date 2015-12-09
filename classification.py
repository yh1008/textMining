__author__      = "Emily Ye Hua"
import time 
import sys
import re
import numpy as np
import nltk
import math
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

reload(sys)  
#csv file needs latin-1 encoding 
sys.setdefaultencoding('latin-1')
stopwords = stopwords.words("english")
#only takes in alpha numerical values
tokenizer = RegexpTokenizer(r'\w+')

print 'Loading Data'
f = open('IMBD_train.csv')

def pre_process (raw_file):
	f = raw_file
	sentiment_score = [] 
	# ( eg. ) review_list = ['Review, Sentiment', 'She likes you', 'He hates you']
	review_list = f.read().splitlines() 
	num_reviews = len(review_list)-1
	corpus_cv = [[] for x in xrange(num_reviews)] #for cross-validation's train_test_split method
	review_list.pop(0)
	corpus = [] #store list with each item the review
	for i  in range(len(review_list)):
		# ( eg.) review_list[i] = 'she likes you' == each review 
		review = review_list[i].rsplit(',', 1)[0]
		review_tokenized= tokenizer.tokenize(review)
		filtered_review = [word for word in review_tokenized if word not in stopwords]
		corpus.append(filtered_review) #['you like me', 'she hate you']		
		sentiment = review_list[i].rsplit(',',1)[1]
		sentiment = int(sentiment)
		sentiment_score.append(sentiment)
		corpus_cv[i] = [review]
		#corpus_cv = [['she likes you'], ['he hate you']]
		#sentiment_score = ['4', '1', '4']
	return corpus, num_reviews, corpus_cv, sentiment_score

#split the test set from train set
def split_train(input_train_reviews, input_train_sentiment):
	train_reviews = input_train_reviews
	train_sentiment = input_train_sentiment
	review_train, review_test, sentiment_train, sentiment_test = train_test_split(train_reviews, train_sentiment, test_size = 0.33, random_state = 22)
	return review_train, review_test, sentiment_train, sentiment_test

def get_train_wordTF(input_train_review_list):
	word_dic = {} 
	review_list = input_train_review_list
	num_reviews = len(review_list)
	i = 0 #index of review
	for entry in review_list:
		review = entry[0]
		review_tokenized= tokenizer.tokenize(review)
		filtered_review = [word for word in review_tokenized if word not in stopwords]
		i += 1
		for word in filtered_review:
			word = word.lower()	
			if word in word_dic:
				word_dic[word] += 1
			else:
				word_dic[word] = 1 
	#word_dic: {"you": 3 }, 3 as the term count 
	return word_dic, num_reviews

def get_test_wordTF(input_train_word_dic, input_test_review_list): 
	i = 0
	train_word_dic = input_train_word_dic
	word_dic = {} 
	sentiment_score = [] 
	review_list = input_test_review_list
	num_reviews = len(review_list)
	for entry in review_list:
		print "test"
		print entry
		review = entry[0]
		review_tokenized= tokenizer.tokenize(review)
		filtered_review = [word for word in review_tokenized if word not in stopwords]
		for word in filtered_review :
			print "filtered_review"
			print word
			if word in train_word_dic: 
				word = word.lower()
				#print word
				if word in word_dic:
					word_dic[word] += 1
				else:
					word_dic[word] = 1 
	#word_dic: {"you": 3 }, 3 as the term count 
	return word_dic, num_reviews, size_dic

def get_IDF(input_word_dic, input_train_review_list, input_test_review_list, flag, input_word_to_index):
	word_to_index = input_word_to_index
	word_dic = input_word_dic
	flag = flag	
	word_to_idf_map = {}
	word_to_docFreq_map = {}
	if flag == "test":
		review_list = input_test_review_list
	else:
		review_list = input_train_review_list
	num_reviews = len(review_list)
	num_reviews = float(num_reviews)
	length = len(word_dic)	
	#log(num_reviews/num_reviews_contains_word)
	for entry in review_list:
		unique_word_per_review = {}
		review = entry[0] #each review
		review_tokenized= tokenizer.tokenize(review)
		filtered_review = [word for word in review_tokenized if word not in stopwords]
		filtered_review = [w.lower() for w in filtered_review]
		for token in filtered_review:
			if flag == "test":
				if token not in word_dic:
					print "unseen words"
					print token
					pass 
				else:
					if token not in unique_word_per_review:
						unique_word_per_review[token] = 0 #value doesn't matter, it is the key 
			else:
				if token not in unique_word_per_review:
					unique_word_per_review[token] = 0
		for unique in unique_word_per_review.keys():
			if unique not in word_to_docFreq_map:
				word_to_docFreq_map[unique] = 1
			else:
				word_to_docFreq_map[unique] += 1 
	for word in word_to_docFreq_map:
		if word_to_docFreq_map[word] >= 1:
			word_to_idf_map[word] =  math.log(float(num_reviews)/word_to_docFreq_map[word])
	return word_to_idf_map


def get_word_to_index(input_word_dic):
	word_dic = input_word_dic
	word_to_index = {}
	key_space = len(word_dic)
	index = 0
	for key in word_dic:
		word_to_index[key] = index
		index += 1 
	return word_to_index

def get_index_to_word(input_word_to_index):
	index_to_word = {}
	word_to_index = input_word_to_index
	index_to_word = {index: word for word, index in word_to_index.items()}
	return index_to_word

def get_bag_of_word(input_review, input_word_to_index, test_or_train, input_train_word_dic, input_num_reviews):
	review_index = 0 
	flag = test_or_train
	train_word_dic = input_train_word_dic
	num_reviews = input_num_reviews
	col = len(train_word_dic)
	bag_of_words = np.zeros((num_reviews, col))
	review_list = input_review
	word_to_index = input_word_to_index
	for entry in review_list:
		review = entry[0]
		review_tokenized= tokenizer.tokenize(review)
		filtered_review = [word for word in review_tokenized if word not in stopwords]
		for word in filtered_review:
			word = word.lower()
			if flag == "test":
			
				if word not in train_word_dic:
					print word
					pass
				else: 
					if bag_of_words[review_index][word_to_index[word]] == 0:
						bag_of_words[review_index][word_to_index[word]] = 1
					else:
						
						bag_of_words[review_index][word_to_index[word]] += 1
			else:
				if bag_of_words[review_index][word_to_index[word]] != 0:
					bag_of_words[review_index][word_to_index[word]] += 1
				else:
					bag_of_words[review_index][word_to_index[word]] = 1
		review_index += 1
	return bag_of_words

def tfidf_weighted_feature(input_bag_of_words, word_to_idf_map, word_to_index, num_reviews, word_dic, size_dic):
	bag_of_words = input_bag_of_words
	size_dic = size_dic
	word_to_idf_map = word_to_idf_map
	num_reviews = num_reviews
	word_dic = word_dic
	col = len(word_dic)
	tfidf = np.zeros((num_reviews, col))
	word_to_index = word_to_index
	for index in range(num_reviews):
		for word in word_to_idf_map:		
			size = np.sum(bag_of_words[index])
			tfidf[index][word_to_index[word]] = (float(bag_of_words[index][word_to_index[word]])/size) * word_to_idf_map[word]
	return tfidf

#random forest/multinomial naive bayes
def train_model(model_name, bag_of_word_feature, sentiment_score):

	model_name = model_name
	if model_name == "randon_forest":
		clf = RandomForestClassifier(n_estimators = 100)
	elif model_name == "multinomial_naive_bayes":
		clf = MultinomialNB()
	model = clf.fit(bag_of_word_feature, sentiment_score)
	return model

def test_model(model, bag_of_word_feature_test):
	model = model
	bag_of_word_feature_test = bag_of_word_feature_test
	prediction = model.predict(bag_of_word_feature_test)
	return prediction

#def plot_confusion_matrix()
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


def main():
	review_list, num_reviews, corpus_cv, sentiment_score = pre_process(f)
	#review_list_test, num_reviews_test, corpus_cv_test, sentiment_score_test = pre_process(f_test)
	review_train, review_test, sentiment_train, sentiment_test = split_train(corpus_cv, sentiment_score)
	word_dic, num_reviews = get_train_wordTF(review_train) 
	word_dic_test, num_reviews_test = get_test_wordTF(word_dic, review_test)
	word_to_index = get_word_to_index(word_dic)
	word_to_index_test = get_word_to_index(word_dic_test)
	index_to_word = get_index_to_word(word_to_index)
	index_to_word_test = get_index_to_word(word_to_index_test)
	bag_of_word_feature = get_bag_of_word(review_train, word_to_index,"train", word_dic, num_reviews)
	bag_of_word_feature_test = get_bag_of_word(review_test, word_to_index, "test", word_dic, num_reviews_test)
	#********************************************************no tfidf weighting******************************
	model = train_model("multinomial_naive_bayes", bag_of_word_feature, sentiment_train)
	prediction = test_model (model, bag_of_word_feature_test)
	print "prediction: "
	print prediction 
	#confusion matrix
	conf_matrix = confusion_matrix(sentiment_test, prediction)
	print "confusion matrix: "
	print conf_matrix
	accuracy = model.score(bag_of_word_feature_test, sentiment_test)
	print "prediction metrics: "
	print 'accuracy: %0.2f'%accuracy
	# Determine the false positive and true positive rates
	prob = model.predict_proba(bag_of_word_feature_test)
	fpr, tpr, _ = roc_curve(sentiment_test, prob[:,1])
	#calculate the AUC
	roc_area = auc(fpr, tpr)
	print 'ROC area: %0.2f' % roc_area

	# Plot of a ROC curve 
	#plot_auc(fpr,tpr,roc_area)
	
	#*****************************************with tfidf weighting******************************************
	IDF_train = get_IDF(word_dic, review_train, review_test, "train", word_to_index)
	IDF_test = get_IDF(word_dic, review_train, review_test, "test", word_to_index)
	tfidf_feature= tfidf_weighted_feature(bag_of_word_feature, IDF_train, word_to_index, num_reviews, word_dic)
	tfidf_feature_test= tfidf_weighted_feature(bag_of_word_feature_test, IDF_test, word_to_index, num_reviews_test, word_dic)
	model_tfidf = train_model("multinomial_naive_bayes", tfidf_feature, sentiment_train)
	prediction = test_model (model, tfidf_feature_test)
	accuracy = model.score(tfidf_feature_test, sentiment_test)
	print 'accuracy: %0.2f'%accuracy
	prob = model_tfidf.predict_proba(bag_of_word_feature_test)
	fpr_1, tpr_1, _1= roc_curve(sentiment_test, prob[:,1])
	#calculate the AUC
	roc_area_1 = auc(fpr_1, tpr_1)
	print 'ROC area: %0.2f' % roc_area_1
	

if __name__ == '__main__':
	time_start = time.clock()
	print "Computing..."
	main()
	time_elapsed = (time.clock() - time_start)
	print "used: %0.2f seconds"%time_elapsed



