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
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
from random import shuffle
import numpy as np
from gensim import utils
import smart_open
from train_doc2vec import EpochLogger
import data_loading
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance

def use_model(model, docs):
    infer_epoch=20
        
    inferred_vectors = []
    for doc in tqdm(docs):
        doc = [w.lower() for w in doc]
        vector = model.infer_vector(doc, alpha=0.01, min_alpha=0.0001, steps=infer_epoch)
        inferred_vectors.append(vector)

    return np.array(inferred_vectors)

def test_core():
    model_path = "data/doc2vec_models/doc2vec_imdb.d2v"

    core_positives = "data/article_perf/core_positive.txt"
    core_negatives = "data/article_perf/core_negative.txt"
    questionable_positives = "data/pos_svm_errors.txt"
    questionable_negatives = "data/neg_svm_errors.txt"

    tsv_file = "data/article_perf/data.tsv"
    metadata_file = "data/article_perf/metadata.tsv"

    model = Doc2Vec.load(model_path)
    with open(core_positives, "r") as f:
        lines = f.readlines()
        vects = use_model(model, lines)
        with open(tsv_file, 'w+') as tensors:
            with open(metadata_file, 'w+') as metadata:
                 metadata.write("tag"+"\t"+"label"+"\n")
                 for idx, emb in enumerate(vects):
                    head_words = " ".join([w.lower() for w in lines[idx].split(" ")][:3])
                    # head_words += '\n'
                    # encoded=head_words.encode('utf-8')
                    metadata.write(head_words+"\t"+"core_pos"+"\n")
                    vector_row = '\t'.join(map(str, emb))
                    tensors.write(vector_row + '\n')

    with open(core_negatives, "r") as f:
        lines = f.readlines()
        vects = use_model(model, lines)
        with open(tsv_file, 'a+') as tensors:
            with open(metadata_file, 'a+') as metadata:
                 for idx, emb in enumerate(vects):
                    head_words = " ".join([w.lower() for w in lines[idx].split(" ")][:3])
                    # head_words += '\n'
                    # encoded=head_words.encode('utf-8')
                    metadata.write(head_words+"\t"+"core_neg"+"\n")
                    vector_row = '\t'.join(map(str, emb))
                    tensors.write(vector_row + '\n')

    # with open(questionable_positives, "r") as f:
    #     lines = f.readlines()
    #     vects = use_model(model, lines)
    #     with open(tsv_file, 'a+') as tensors:
    #         with open(metadata_file, 'a+') as metadata:
    #              for idx, emb in enumerate(vects):
    #                 head_words = " ".join([w.lower() for w in lines[idx].split(" ")][:3])
    #                 # head_words += '\n'
    #                 # encoded=head_words.encode('utf-8')
    #                 metadata.write(head_words+"\t"+"quest_pos"+"\n")
    #                 vector_row = '\t'.join(map(str, emb))
    #                 tensors.write(vector_row + '\n')

    # with open(questionable_negatives, "r") as f:
    #     lines = f.readlines()
    #     vects = use_model(model, lines)
    #     with open(tsv_file, 'a+') as tensors:
    #         with open(metadata_file, 'a+') as metadata:
    #              for idx, emb in enumerate(vects):
    #                 head_words = " ".join([w.lower() for w in lines[idx].split(" ")][:3])
    #                 # head_words += '\n'
    #                 # encoded=head_words.encode('utf-8')
    #                 metadata.write(head_words+"\t"+"quest_neg"+"\n")
    #                 vector_row = '\t'.join(map(str, emb))
    #                 tensors.write(vector_row + '\n')

# def test_triplets():
#     triplets = [["good", "like", "dislike"],
#                 ["good", "amazing", "bad"],
#                 ["enjoyed", "nice", "dislike"],
#                 ["interesting", "intriguing", "boring"],
#                 ["marvelous", "worth", "wasted"],
#                 ["worth", "would recommend", "would not recommend"],
#                 ["i liked", "i enjoyed", "i disliked",
#                 ["it was boring", "it wasted my time", "i would watch it again"]]

def test_docs():
    positive_lexicon = []
    negative_lexicon = []

    with open("data/pos_sent_words.txt", "r") as f:
        lines = f.readlines()
        positive_lexicon = lines

    with open("data/neg_sent_words.txt", "r") as f:
        lines = f.readlines()
        negative_lexicon = lines

    negative_lexicon = [w.strip() for w in negative_lexicon]
    positive_lexicon = [w.strip() for w in positive_lexicon]

    model_path = "data/doc2vec_models/baseline_window_10.d2v"
    model = Doc2Vec.load(model_path)
    train_pos_path = "data/data-tagged/POS/"
    train_neg_path = "data/data-tagged/NEG/"
    pos_train, pos_val, pos_test, neg_train, neg_val, neg_test = data_loading.load_data_kfold_10_test_val(train_pos_path, train_neg_path, ["\n"], 1, 0)
    pos_vectors = use_model(model, pos_train)
    neg_vectors = use_model(model, neg_train)
    pos_vectors =list(pos_vectors)
    neg_vectors = list(neg_vectors)

    error_positive_docs = 0
    error_negative_docs = 0
    correct_positive_docs = 0
    correct_negative_docs = 0
    for idx, pos_train_doc in enumerate(pos_vectors):
        if idx < len(pos_vectors)-1:
            #distance_similar = dot(pos_train_doc, pos_vectors[idx+1])/(norm(pos_train_doc)*norm(pos_vectors[idx+1]))
            #distance_not_similar = dot(pos_train_doc, neg_vectors[idx+1])/(norm(pos_train_doc)*norm(neg_vectors[idx+1]))
            distance_similar = distance.cosine(pos_train_doc, pos_vectors[idx+1])
            distance_not_similar = distance.cosine(pos_train_doc, neg_vectors[idx])
            if distance_similar > distance_not_similar:
                error_positive_docs += 1
            else:
                correct_positive_docs += 1

    for idx, neg_train_doc in enumerate(neg_vectors):
        if idx < len(neg_vectors)-1:
            #distance_similar = dot(pos_train_doc, pos_vectors[idx+1])/(norm(pos_train_doc)*norm(pos_vectors[idx+1]))
            #distance_not_similar = dot(pos_train_doc, neg_vectors[idx+1])/(norm(pos_train_doc)*norm(neg_vectors[idx+1]))
            distance_similar = distance.cosine(neg_train_doc, neg_vectors[idx+1])
            distance_not_similar = distance.cosine(neg_train_doc, pos_vectors[idx])
            if distance_similar > distance_not_similar:
                error_negative_docs += 1
            else:
                correct_negative_docs += 1

    accuracy = (correct_positive_docs+correct_negative_docs)/(error_negative_docs+correct_positive_docs+correct_negative_docs+error_positive_docs)
    print(accuracy)
    print(error_positive_docs)
    print(error_negative_docs)
    print(correct_positive_docs)
    print(correct_negative_docs)

def test_perfect_pos_neg():
    positive_lexicon = []
    negative_lexicon = []

    with open("data/pos_sent_words.txt", "r") as f:
        lines = f.readlines()
        positive_lexicon = lines

    with open("data/neg_sent_words.txt", "r") as f:
        lines = f.readlines()
        negative_lexicon = lines

    negative_lexicon = [w.strip() for w in negative_lexicon]
    positive_lexicon = [w.strip() for w in positive_lexicon]

    model_path = "data/doc2vec_models/baseline_window_10.d2v"
    model = Doc2Vec.load(model_path)
    train_pos_path = "data/data-tagged/POS/"
    train_neg_path = "data/data-tagged/NEG/"
    pos_train, pos_val, pos_test, neg_train, neg_val, neg_test = data_loading.load_data_kfold_10_test_val(train_pos_path, train_neg_path, ["\n"], 1, 0)
    pos_vectors = use_model(model, pos_train)
    neg_vectors = use_model(model, neg_train)
    pos_vectors =list(pos_vectors)
    neg_vectors = list(neg_vectors)

    perfect_pos = model.infer_vector(positive_lexicon, alpha=0.01, min_alpha=0.0001, steps=20)
    perfect_neg = model.infer_vector(negative_lexicon, alpha=0.01, min_alpha=0.0001, steps=20)

    error_positive_docs = 0
    error_negative_docs = 0
    correct_positive_docs = 0
    correct_negative_docs = 0
    for idx, pos_train_doc in enumerate(pos_vectors):
        if idx < len(pos_vectors)-1:
            #distance_similar = dot(pos_train_doc, pos_vectors[idx+1])/(norm(pos_train_doc)*norm(pos_vectors[idx+1]))
            #distance_not_similar = dot(pos_train_doc, neg_vectors[idx+1])/(norm(pos_train_doc)*norm(neg_vectors[idx+1]))
            distance_similar = distance.cosine(pos_train_doc, perfect_pos)
            distance_not_similar = distance.cosine(pos_train_doc, perfect_neg)
            if distance_similar > distance_not_similar:
                error_positive_docs += 1
            else:
                correct_positive_docs += 1

    for idx, neg_train_doc in enumerate(neg_vectors):
        if idx < len(neg_vectors)-1:
            #distance_similar = dot(pos_train_doc, pos_vectors[idx+1])/(norm(pos_train_doc)*norm(pos_vectors[idx+1]))
            #distance_not_similar = dot(pos_train_doc, neg_vectors[idx+1])/(norm(pos_train_doc)*norm(neg_vectors[idx+1]))
            distance_similar = distance.cosine(neg_train_doc, perfect_neg)
            distance_not_similar = distance.cosine(neg_train_doc, perfect_pos)
            if distance_similar > distance_not_similar:
                error_negative_docs += 1
            else:
                correct_negative_docs += 1

    accuracy = (correct_positive_docs+correct_negative_docs)/(error_negative_docs+correct_positive_docs+correct_negative_docs+error_positive_docs)
    print(accuracy)
    print(error_positive_docs)
    print(error_negative_docs)
    print(correct_positive_docs)
    print(correct_negative_docs)

test_docs()
test_perfect_pos_neg()

