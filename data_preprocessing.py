from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm
import collections
import operator


def tokenize_text(text):
	words = re.sub("[^\w]", " ",  text).split()
	tokenized = [w.lower() for w in words]
	return tokenized

# Initial implementation for unigram
def generate_embeddings(train_pos_data_path, train_neg_data_path):
	all_words = []
	tokenized_docs = []

	train_files = [f for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
	
	for train_file in tqdm(train_files):
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

	# Check from where to start
	if max_grams != 1:
		if min_grams == 1:

			# If the unigrams are included in the requested embeddings include them in the list
			diff_grams.append(old_tokens)

			min_grams += 1

		else:
			# Else skip unigrams
			tokens = []

		orig_tokens_len = len(old_tokens)

		for n in range(min_grams, min(max_grams + 1, orig_tokens_len+1)):
			
			#Create a list that will store the concrete values of the n_gram calculation
			n_grams = []

			for i in range(orig_tokens_len - n + 1):

				# Append the concrete n_gram value to the embedding of the document
				n_gram = " ".join(old_tokens[i:i+n])

				# Append the n-gram embeddings to the general embeddings of the document
				tokens.append(n_gram)

				# Append the n-grams to a different list, so each n-grams can be filtered separately
				n_grams.append(n_gram)

			diff_grams.append(n_grams)

	return tokens, diff_grams


def generate_embeddings_generic(min_grams, max_grams, train_files):
	tokenized_docs = []
	unigrams = []
	bigrams = []

	print("All files are: ", len(train_files))
	for tokenized_text in tqdm(train_files):
		tokens, diff_grams = generate_n_grams(min_grams, max_grams, tokenized_text)

		tokenized_docs.append(tokens)

		unigrams.extend(diff_grams[0])
		bigrams.extend(diff_grams[1])

	vocab = []
	unigrams = dict(collections.Counter(unigrams))

	# Filter the unigrams, so that only the ones that occur more than 4 times are left in the vocabulary
	for (key,value) in unigrams.items():
		# Check if an item occurs more than 4 times
		if value >= 4:
			vocab.append(key)


	# Take as many bigrams as many unigrams are there in the vocabulary 
	# NOTE THAT THIS IS NOT EXPLICITLY MENTIONED IN THE ARTICLE
	len_all_unigrams =len(vocab)

	bigrams = dict(collections.Counter(bigrams))
	sorted_bigrams = sorted(bigrams.items(), key=operator.itemgetter(1), reverse=True)
	
	for i, item in enumerate(sorted_bigrams):
		vocab.append(item[0])

		if i > len_all_unigrams:
			break
	vocab = sorted(vocab)

	return vocab, tokenized_docs

	# bag_vectors = []

	# for doc in tqdm(tokenized_docs):
	# 	bag = [0] * len(vocab)

	# 	for w in doc:
	# 		for i, word in enumerate(vocab):
	# 			if word == w:
	# 				# If we need also the count, it can be bag[i] += 1 
	# 				bag[i] = 1

	# 	bag_vectors.append(bag)

	# print(bag_vectors[0])

