from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
import collections
import operator

def collect_train_data(train_data_files):
	all_docs = []
	for file in tqdm(train_data_files):
		with open(file, "r") as f:
			text = f.readlines()
			all_tokens = []
			for token in text:
				token = token.split("\t")
				all_tokens.append(token[0])

			# Removing the new line token
			for i, token in enumerate(all_tokens):
				if token == "\n":
					del all_tokens[i]

		all_docs.append(all_tokens)

	return all_docs

def load_data(train_pos_data_path, train_neg_data_path):

	train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

	train_files_pos.sort(key = lambda x: x.split("_")[0].replace("cv", ""))
	train_files_neg.sort(key = lambda x: x.split("_")[0].replace("cv", ""))

	print("The count of all positive files is: ", len(train_files_pos))
	print("The count of all negative files is: ", len(train_files_neg))

	all_pos_docs = collect_train_data(train_files_pos)
	all_neg_docs = collect_train_data(train_files_neg)

	# Split train and test data
	pos_train = all_pos_docs[0:899]
	pos_test = all_pos_docs[899:]

	neg_train = all_neg_docs[0:899]
	neg_test = all_neg_docs[899:]

	return pos_train, pos_test, neg_train, neg_test

