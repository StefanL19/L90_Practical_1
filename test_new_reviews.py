from sklearn import svm, linear_model
import pandas as pd
from tqdm import tqdm
import metrics
import numpy as np
import math
from sklearn.externals import joblib
import locale
import glob
import os.path
import requests
import tarfile
import sys
import codecs
import smart_open 
import re
import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import LabeledSentence
from tqdm import tqdm
from train_doc2vec import use_model, EpochLogger, EpochSaver

new_positive_reviews = "data/2018_data/positive_reviews.txt"
new_negative_reviews = "data/2018_data/negative_reviews.txt"
doc2vec_path = "data/doc2vec_models/final_10.d2v"

with open(new_positive_reviews, "r") as f:
	lines = f.readlines()
	lines = [l.strip() for l in lines]
	pos_reviews = lines

with open(new_negative_reviews, "r") as f:
	lines = f.readlines()
	lines = [l.strip() for l in lines]
	neg_reviews = lines

print(pos_reviews)
print(neg_reviews)

gt = [1]*10 + [0]*10

pos_embeddings = use_model(doc2vec_path, pos_reviews)
neg_embeddings = use_model(doc2vec_path, neg_reviews)

X = np.vstack((pos_embeddings, neg_embeddings))
Y = np.array(gt)

clf = joblib.load('data/model_repo/svm_test.pkl')
predictions = clf.predict(X)
print(predictions)
