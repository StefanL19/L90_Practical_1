
#  In Python, an individual
#  * document should look like this:
#  * >> (<label>, [(<feature>, <value>), ...]) */
# static int unpack_document(

import data_loading
import data_preprocessing
from tqdm import tqdm
import pandas as pd
import multiprocessing
from classifiers import count_word_occurences
from functools import partial 
import numpy as np

def convert_embeddings_to_svm_format(embeddings, doc_class):
   """
      embeddings: a list of vectors
   """

   embeddings_formatted = []

   for embedding in embeddings:
      embedding_formatted = (doc_class, [])
      for idx, feature_value in enumerate(embedding):
         
         # Make sure that the number of occurences does not influence the embedding
         if feature_value > 1:
            feature_value = 1

         embedding_formatted[1].append((idx, feature_value))

      embeddings_formatted.append(embedding_formatted)

   return embeddings_formatted


def bound_array(a):
   for idx, el in enumerate(a):
      if el > 1.:
         a[idx] = 1.

   return a

def preprocess_data(train_pos_path, train_neg_path, stopwords, VAL_CATEGORY, TEST_CATEGORY):
   general_path = "data/svm_bow/"+str(val_cat)+"/"

   import os
   if not os.path.exists(general_path+"train/"):
      os.makedirs(general_path+"train/")

   if not os.path.exists(general_path+"val/"):
      os.makedirs(general_path+"val/")

   if not os.path.exists(general_path+"test/"):
      os.makedirs(general_path+"test/")

   # Preprocess and split the data into train and test sets
   pos_train, pos_val, pos_test, neg_train, neg_val, neg_test = data_loading.load_data_kfold_10_test_val(train_pos_path, train_neg_path, stopwords, VAL_CATEGORY, TEST_CATEGORY)

   # Files that will be used for training
   train_files = pos_train + neg_train

   # Generate n-grams for embeddings
   vocab, docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, train_files)
   pos_docs_tokenized_train = docs_tokenized[:len(pos_train)]
   neg_docs_tokenized_train = docs_tokenized[len(pos_train):]

   m = multiprocessing.Manager()
   pos_list = m.list()
   neg_list = m.list()
   pos_list_test = m.list()
   neg_list_test = m.list()
   pos_list_validation = m.list()
   neg_list_validation = m.list()

   vocab_length = len(vocab)

   print("Started iterating training positive documents")
   with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
      pool.map(partial(count_word_occurences, vocab_length, pos_list, vocab), pos_docs_tokenized_train)

   print("Started iterating training negative documents")
   with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
      pool.map(partial(count_word_occurences, vocab_length, neg_list, vocab), neg_docs_tokenized_train)

   # Generate training docs SVM Embeddings and save them to a dataframe in a csv file
   pos_list_bound = [bound_array(l) for l in pos_list]
   pos_list_bound = np.array(pos_list_bound)
   pos_train_path = 'pos_train_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
   path = general_path+"train/"+pos_train_path
   np.savetxt(path, pos_list_bound)

   neg_list_bound = [bound_array(l) for l in neg_list]
   neg_list_bound = np.array(neg_list_bound)
   neg_train_df = pd.DataFrame(neg_list_bound)
   neg_train_path = 'neg_train_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
   path = general_path+"train/"+neg_train_path
   np.savetxt(path, neg_list_bound)




   # # Files that will be used for validation
   # val_files = pos_val + neg_val

   # _, val_docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, val_files)
   # pos_docs_tokenized_val = val_docs_tokenized[:len(pos_val)]
   # neg_docs_tokenized_val = val_docs_tokenized[len(neg_val):]

   # print("Started iterating validation positive documents")
   # with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
   #    pool.map(partial(count_word_occurences, vocab_length, pos_list_validation, vocab), pos_docs_tokenized_val)

   # print("Started iterating validation negative documents")
   # with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
   #    pool.map(partial(count_word_occurences, vocab_length, neg_list_validation, vocab), neg_docs_tokenized_val)

   # # Generate validation docs SVM Embeddings and save them to a dataframe in a csv file
   # pos_val_embeddings_format = convert_embeddings_to_svm_format(pos_list_validation, 1)
   # pos_val_df = pd.DataFrame(pos_val_embeddings_format)

   # pos_val_path = 'pos_val_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
   # pos_val_df.to_csv(general_path+"val/"+pos_val_path, index=False, header=False)

   # neg_val_embeddings_format = convert_embeddings_to_svm_format(neg_list_validation, 0)
   # neg_val_df = pd.DataFrame(neg_val_embeddings_format)

   # neg_val_path = 'neg_val_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
   # neg_val_df.to_csv(general_path+"val/"+neg_val_path, index=False, header=False)




   # # Files that will be used for testing
   # test_files = pos_test + neg_test

   # _, test_docs_tokenized = data_preprocessing.generate_embeddings_generic(1, 2, test_files)
   # pos_docs_tokenized_test = test_docs_tokenized[:len(pos_test)]
   # neg_docs_tokenized_test = test_docs_tokenized[len(pos_test):]

   # print("Started iterating test positive documents")
   # with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
   #    pool.map(partial(count_word_occurences, vocab_length, pos_list_test, vocab), pos_docs_tokenized_test)

   # print("Started iterating test negative documents")
   # with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
   #    pool.map(partial(count_word_occurences, vocab_length, neg_list_test, vocab), neg_docs_tokenized_test)

   # # Generate validation docs SVM Embeddings and save them to a dataframe in a csv file
   # pos_test_embeddings_format = convert_embeddings_to_svm_format(pos_list_test, 1)
   # pos_test_df = pd.DataFrame(pos_test_embeddings_format)
   # pos_test_path = 'pos_test_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
   # pos_test_df.to_csv(general_path+"test/"+pos_test_path, index=False, header=False)

   # neg_test_embeddings_format = convert_embeddings_to_svm_format(neg_list_test, 0)
   # neg_test_df = pd.DataFrame(neg_test_embeddings_format)
   # neg_test_path = 'neg_test_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
   # neg_test_df.to_csv(general_path +"test/"+ neg_test_path, index=False, header=False)

train_pos_path = "data/data-tagged/POS/"
train_neg_path = "data/data-tagged/NEG/"
stopwords = ["\n"]

# Define Validation and Test Categories
VAL_CATEGORY = 1
TEST_CATEGORY = 0

for val_cat in range(2,10):
   print("Working on validation category ", val_cat)
   preprocess_data(train_pos_path, train_neg_path, stopwords, val_cat, TEST_CATEGORY)


# pos_tr = pd.read_csv("data/pos_train_val_1_test_0.csv", header=None)
# print(pos_tr.values[0][0])

