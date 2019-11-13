from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
import collections
import operator
import numpy as np

def collect_train_data(train_data_files, stopwords):
	"""
		Reads the files and tokenizes the entries in them by also removing the stopwords
		train_data_files: The paths for the files that should be tokenized
		stopwords: List of stopwords that should be excluded
	"""
	all_docs = []
	for file in tqdm(train_data_files):
		with open(file, "r") as f:
			text = f.readlines()
			all_tokens = []
			for token in text:
				token = token.split("\t")

				#Append token to the list by making it appear in lower case
				all_tokens.append(token[0])

			# Removing the new line token
			for i, token in enumerate(all_tokens):
				if token in stopwords:
					del all_tokens[i]

		all_docs.append(all_tokens)

	return all_docs

def load_data_kfold_10_test_val(train_pos_data_path, train_neg_data_path, stopwords, val_category, test_category):
	"""
		Reads the data, tokenizes it by removing the stopwords and splits it by taking 9 out of 10 folds for training and 1 fold for testing
	"""

	train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

	train_files_pos.sort(key = lambda x: x.split("_")[0].replace("cv", ""))
	train_files_neg.sort(key = lambda x: x.split("_")[0].replace("cv", ""))

	print("The count of all positive files is: ", len(train_files_pos))
	print("The count of all negative files is: ", len(train_files_neg))

	all_pos_docs = collect_train_data(train_files_pos, stopwords)
	all_neg_docs = collect_train_data(train_files_neg, stopwords)

	pos_train = []
	neg_train = []
	pos_val = []
	neg_val = []
	pos_test = []
	neg_test = []
	remove_pos = []
	remove_neg = []

	for idx, pos_entry in enumerate(all_pos_docs):
		# If the index falls into the test category
		if (idx%10) == test_category:
			# Add the entry to the positive test set
			pos_test.append(pos_entry)
		elif (idx%10) == val_category:
			# Add the entry to the positive validation set
			pos_val.append(pos_entry)
		else:
			# Add the entry to the positive train set
			pos_train.append(pos_entry)


	# Do the same for the negative set
	for idx, neg_entry in enumerate(all_neg_docs):
		# If the index falls into the test category
		if (idx%10) == test_category:
			# Add the entry to the negative test set
			neg_test.append(neg_entry)
		elif (idx%10) == val_category:
			# Add the entry to the negative validation set
			neg_val.append(neg_entry)
		else:
			neg_train.append(neg_entry)

	print("The size of the positive training set is: ", len(pos_train))
	print("The size of the negative training set is: ", len(neg_train))
	print("The size of the positive validation set is: ", len(pos_val))
	print("The size o the negative validation set is: ", len(neg_val))
	print("The size of the positive test set is: ", len(pos_test))
	print("The size of the negative test set is: ", len(neg_test))

	return pos_train, pos_val, pos_test, neg_train, neg_val, neg_test

def load_data_kfold_10_test(train_pos_data_path, train_neg_data_path, stopwords, test_category):
	"""
		Reads the data, tokenizes it by removing the stopwords and splits it by taking 9 out of 10 folds for training and 1 fold for testing
	"""

	train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

	train_files_pos.sort(key = lambda x: x.split("_")[0].replace("cv", ""))
	train_files_neg.sort(key = lambda x: x.split("_")[0].replace("cv", ""))

	print("The count of all positive files is: ", len(train_files_pos))
	print("The count of all negative files is: ", len(train_files_neg))

	all_pos_docs = collect_train_data(train_files_pos, stopwords)
	all_neg_docs = collect_train_data(train_files_neg, stopwords)

	pos_train = []
	neg_train = []
	pos_test = []
	neg_test = []
	remove_pos = []
	remove_neg = []

	for idx, pos_entry in enumerate(all_pos_docs):
		# If the index falls into the test category
		if (idx%10) == test_category:
			# Add the entry to the positive test set
			pos_test.append(pos_entry)
		else:
			# Add the entry to the positive train set
			pos_train.append(pos_entry)


	# Do the same for the negative set
	for idx, neg_entry in enumerate(all_neg_docs):
		# If the index falls into the test category
		if (idx%10) == test_category:
			# Add the entry to the negative test set
			neg_test.append(neg_entry)
		else:
			neg_train.append(neg_entry)

	print("The size of the positive training set is: ", len(pos_train))
	print("The size of the negative training set is: ", len(neg_train))
	print("The size of the positive test set is: ", len(pos_test))
	print("The size of the negative test set is: ", len(neg_test))

	return pos_train, pos_test, neg_train, neg_test


def load_data_simple_train_test(train_pos_data_path, train_neg_data_path, stopwords):
	"""
		Reads the data, tokenizes it by removing the stopwords and splits it by taking 9 out of 10 folds for training and 1 fold for testing
	"""

	train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

	train_files_pos.sort(key = lambda x: x.split("_")[0].replace("cv", ""))
	train_files_neg.sort(key = lambda x: x.split("_")[0].replace("cv", ""))

	print("The count of all positive files is: ", len(train_files_pos))
	print("The count of all negative files is: ", len(train_files_neg))

	all_pos_docs = collect_train_data(train_files_pos, stopwords)
	all_neg_docs = collect_train_data(train_files_neg, stopwords)

	pos_train = all_pos_docs[:900]
	neg_train = all_neg_docs[:900]
	pos_test = all_pos_docs[900:]
	neg_test = all_neg_docs[900:]

	print("The size of the positive training set is: ", len(pos_train))
	print("The size of the negative training set is: ", len(neg_train))
	print("The size of the positive test set is: ", len(pos_test))
	print("The size of the negative test set is: ", len(neg_test))

	return pos_train, pos_test, neg_train, neg_test

