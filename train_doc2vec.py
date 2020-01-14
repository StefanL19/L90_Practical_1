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

# import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing
from random import shuffle
import numpy as np
from gensim import utils
import smart_open

# from collections import namedtuple

# # this data object class suffices as a `TaggedDocument` (with `words` and `tags`) 
# # plus adds other state helpful for our later evaluation/reporting
# SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            with smart_open.open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])
    
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with smart_open.open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences
    
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

def train_proper():
    sources = {'data/aclImdb/test-neg.txt':'TEST_NEG', 'data/aclImdb/test-pos.txt':'TEST_POS', 'data/aclImdb/train-neg.txt':'TRAIN_NEG', 'data/aclImdb/train-pos.txt':'TRAIN_POS', 'data/aclImdb/train-unsup.txt':'TRAIN_UNS'}

    sentences = LabeledLineSentence(sources)

    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)

    model.build_vocab(sentences.to_array())

    for epoch in range(10):
        print("Iterating epoch: {}".format(epoch))
        model.train(sentences.sentences_perm(), total_examples=model.corpus_count, epochs=1)

    model.save('data/imdb.d2v')

class EpochLogger(CallbackAny2Vec):
     '''Callback to log information about training'''

     def __init__(self):
         self.epoch = 0

     def on_epoch_begin(self, model):
         print("Epoch #{} start".format(self.epoch))

     def on_epoch_end(self, model):
         print("Epoch #{} end".format(self.epoch))
         self.epoch += 1
        
class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, path_prefix):
     self.path_prefix = path_prefix
     self.epoch = 0

    def on_epoch_end(self, model):
     output_path = 'data/doc2vec_models/{}_epoch{}.model'.format(self.path_prefix, self.epoch)
     model.save(output_path)
     self.epoch += 1

def train_model():
    alldocs = []

    with smart_open.smart_open('data/aclImdb/alldata-id.txt', 'rb', encoding='utf-8') as alldata:
        for line_no, line in enumerate(alldata):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            tags = [line_no] # 'tags = [tokens[0]]' would also work at extra memory cost
            alldocs.append(TaggedDocument(words=words, tags=tags))

    print("All docs that will be used for training are: ", len(alldocs))

    doc_list = alldocs[:]  
    shuffle(doc_list)

    cores = multiprocessing.cpu_count() - 1

    model = Doc2Vec(dm=0, vector_size=200, window=10, hs=1, min_count=7, sample=1e-1, 
                epochs=30, workers=cores, alpha=0.025, alpha_min=0.0001)

    model.build_vocab(alldocs)
    epoch_logger = EpochLogger()
    epoch_saver = EpochSaver(path_prefix="init_model")

    model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs, callbacks=[epoch_logger])
    model.save("data/doc2vec_models/baseline_hm.d2v")

def use_model(model_path, docs):
    infer_epoch=20
    model = Doc2Vec.load(model_path)
        
    inferred_vectors = []
    for doc in tqdm(docs):
        doc = [w.lower() for w in doc]
        #doc = [w.lower() for w in doc if (w in positive_lexicon) or (w in negative_lexicon)]
        vector = model.infer_vector(doc, alpha=0.01, min_alpha=0.0001, steps=infer_epoch)
        inferred_vectors.append(vector)

    return np.array(inferred_vectors)


if __name__ == "__main__":
    train_model()