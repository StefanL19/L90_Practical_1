from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
import collections

def tokenize_text(text):
	words = re.sub("[^\w]", " ",  text).split()
	tokenized = [w.lower() for w in words]
	return tokenized

# Initial implementation for unigram
def generate_embeddings(train_pos_data_path, train_neg_data_path):
	all_words = []
	tokenized_docs = []

	train_files = [f for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	
	for train_file in tqdm(train_files[:1000]):
		file_full_path = train_pos_data_path+"/"+train_file
		with open(file_full_path) as f:
			text = f.read()
			tokenized_text = tokenize_text(text)
			tokenized_docs.append(tokenized_text)

			all_words.extend(tokenized_text)

	# Remove all words that repeat
	vocab = sorted(list(set(all_words)))
	
	print("There are {0} document with {1} unique unigrams".format(len(tokenized_docs), len(vocab)))

	bag_vectors = []

	for doc in tqdm(tokenized_docs):
		bag = [0] * len(vocab)

		for w in doc:
			for i, word in enumerate(vocab):
				if word == w:
					bag[i] += 1

		bag_vectors.append(bag)


def generate_n_grams(min_grams, max_grams, tokens):
	old_tokens = tokens.copy()

	# Create a list of different n_grams occuring in the doc

	diff_grams = []

	if max_grams != 1:
		if min_grams == 1:
			diff_grams.append(old_tokens)

			min_grams += 1

		else:
			tokens = []

		orig_tokens_len = len(old_tokens)

		for n in range(min_grams, min(max_grams + 1, orig_tokens_len+1)):
			#Create a list that will store the concrete values of the n_gram calculation
			n_grams = []

			for i in range(orig_tokens_len - n + 1):
				# Append the concrete n_gram value to the embedding of the document
				n_gram = " ".join(old_tokens[i:i+n])
				tokens.append(n_gram)
				n_grams.append(n_gram)

			diff_grams.append(n_grams)

	return tokens, diff_grams


def generate_embeddings_generic(min_grams, max_grams, train_pos_data_path, train_neg_data_path):
	#all_words = []
	tokenized_docs = []
	unigrams = []
	bigrams = []

	train_files = [f for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]

	for train_file in tqdm(train_files[0:1]):
		file_full_path = train_pos_data_path+"/"+train_file

		with open(file_full_path) as f:
			text = f.read()
			tokenized_text = tokenize_text(text)
			tokens, diff_grams = generate_n_grams(min_grams, max_grams, tokenized_text)
			tokenized_docs.append(tokens)

			unigrams.extend(diff_grams[0])
			bigrams.extend(diff_grams[1])

	#print(collections.Counter(bigrams).most_common())


	#vocab = print(collections.Counter(all_words).most_common())

	#print(set(all_words))
	# # Remove all words that repeat
	# vocab = sorted(list(set(all_words)))
	
	# print("There are {0} document with {1} unique unigrams and bigrams".format(len(tokenized_docs), len(vocab)))

	# bag_vectors = []

	# for doc in tqdm(tokenized_docs):
	# 	bag = [0] * len(vocab)

	# 	for w in doc:
	# 		for i, word in enumerate(vocab):
	# 			if word == w:
	# 				bag[i] += 1

	# 	bag_vectors.append(bag)

	# print(bag_vectors[0])

