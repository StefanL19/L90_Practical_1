import data_loading
import data_preprocessing
from tqdm import tqdm
import pandas as pd
import multiprocessing
from classifiers import count_word_occurences
from functools import partial 
import numpy as np
from train_doc2vec import use_model, EpochSaver, EpochLogger

def preprocess_big_data():
    train_pos_data_path = "data/aclImdb/train-pos.txt"
    train_neg_data_path = "data/aclImdb/train-neg.txt"

    train_files_pos = [join(train_pos_data_path, f) for f in listdir(train_pos_data_path) if isfile(join(train_pos_data_path, f))]
    train_files_neg = [join(train_neg_data_path, f) for f in listdir(train_neg_data_path) if isfile(join(train_neg_data_path, f))]

    

def preprocess_data(train_pos_path, train_neg_path, stopwords, model_path, VAL_CATEGORY, TEST_CATEGORY):
    with open("data/pos_sent_words.txt", "r") as f:
        lines = f.readlines()
        positive_lexicon = lines

    with open("data/neg_sent_words.txt", "r") as f:
        lines = f.readlines()
        negative_lexicon = lines

    negative_lexicon = [w.strip() for w in negative_lexicon]
    positive_lexicon = [w.strip() for w in positive_lexicon]

    general_path = "data/svm_doc2vec/"+str(VAL_CATEGORY)+"/"

    import os
    if not os.path.exists(general_path+"train/"):
      os.makedirs(general_path+"train/")

    if not os.path.exists(general_path+"val/"):
      os.makedirs(general_path+"val/")

    if not os.path.exists(general_path+"test/"):
      os.makedirs(general_path+"test/")


    pos_train, pos_val, pos_test, neg_train, neg_val, neg_test = data_loading.load_data_kfold_10_test_val(train_pos_path, train_neg_path, stopwords, VAL_CATEGORY, TEST_CATEGORY)

    print("Saving the validation set: ")
    with open(general_path+"val/val_docs_pos.txt", "a") as f:
        for doc in pos_val:
            doc_str = " ".join(doc)
            f.write(doc_str)
            f.write("\n")

    with open(general_path+"val/val_docs_neg.txt", "a") as f:
        for doc in neg_val:
            doc_str = " ".join(doc)
            f.write(doc_str)
            f.write("\n")


    # Training embeddings
    pos_train_embeddings = use_model(model_path, pos_train)
    neg_train_embeddings = use_model(model_path, neg_train)

    pos_train_path = 'pos_train_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
    neg_train_path = 'neg_train_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)

    np.savetxt(general_path+"train/"+pos_train_path, pos_train_embeddings)
    np.savetxt(general_path+"train/"+neg_train_path, neg_train_embeddings)

    # Validation Embeddings
    pos_val_embeddings = use_model(model_path, pos_val)
    neg_val_embeddings = use_model(model_path, neg_val)

    pos_val_path = 'pos_val_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
    neg_val_path = 'neg_val_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)

    np.savetxt(general_path+"val/"+pos_val_path, pos_val_embeddings)
    np.savetxt(general_path+"val/"+neg_val_path, neg_val_embeddings)

    # Test Embeddings
    pos_test_embeddings = use_model(model_path, pos_test)
    neg_test_embeddings = use_model(model_path, neg_test)

    pos_test_path = 'pos_test_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)
    neg_test_path = 'neg_test_val_%d_test_%d.csv' %(VAL_CATEGORY, TEST_CATEGORY)

    np.savetxt(general_path+"test/"+pos_test_path, pos_test_embeddings)
    np.savetxt(general_path+"test/"+neg_test_path, neg_test_embeddings)

train_pos_path = "data/data-tagged/POS/"
train_neg_path = "data/data-tagged/NEG/"
stopwords = ["\n"]

# Define Validation and Test Categories
TEST_CATEGORY = 0
MODEL_PATH = "data/doc2vec_models/final_10.d2v"
for val_cat in range(1,10):
   print("Working on validation category ", val_cat)
   preprocess_data(train_pos_path, train_neg_path, stopwords, MODEL_PATH, val_cat, TEST_CATEGORY)
   