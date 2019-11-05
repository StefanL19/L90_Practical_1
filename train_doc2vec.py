import locale
import glob
import os.path
import requests
import tarfile
import sys
import codecs
from smart_open import smart_open
import re
import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile

# dirname = 'data/aclImdb'
# filename = 'aclImdb_v1.tar.gz'
# locale.setlocale(locale.LC_ALL, 'C')
# all_lines = []

# if sys.version > '3':
#     control_chars = [chr(0x85)]
# else:
#     control_chars = [unichr(0x85)]

# # Convert text to lower-case and strip punctuation/symbols from words
# def normalize_text(text):
#     norm_text = text.lower()
#     # Replace breaks with spaces
#     norm_text = norm_text.replace('<br />', ' ')
#     # Pad punctuation with spaces on both sides
#     norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
#     return norm_text

# def prepare_docs():
# 	folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']

# 	for fol in folders:
# 	        temp = u''
# 	        newline = "\n".encode("utf-8")
# 	        output = fol.replace('/', '-') + '.txt'
# 	        # Is there a better pattern to use?
# 	        txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
# 	        print(" %s: %i files" % (fol, len(txt_files)))
# 	        with smart_open(os.path.join(dirname, output), "wb") as n:
# 	            for i, txt in enumerate(txt_files):
# 	                with smart_open(txt, "rb") as t:
# 	                    one_text = t.read().decode("utf-8")
# 	                    for c in control_chars:
# 	                        one_text = one_text.replace(c, ' ')
# 	                    one_text = normalize_text(one_text)
# 	                    all_lines.append(one_text)
# 	                    n.write(one_text.encode("utf-8"))
# 	                    n.write(newline)

# 	# Save to disk for instant re-use on any future runs
# 	with smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
# 	    for idx, line in enumerate(all_lines):
# 	        num_line = u"_*{0} {1}\n".format(idx, line)
# 	        f.write(num_line.encode("utf-8"))

# 	assert os.path.isfile("data/aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"
# 	print("Success, alldata-id.txt is available for next steps.")

# import gensim
from gensim.models.doc2vec import TaggedDocument
# from collections import namedtuple

# # this data object class suffices as a `TaggedDocument` (with `words` and `tags`) 
# # plus adds other state helpful for our later evaluation/reporting
# SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

class EpochLogger(CallbackAny2Vec):
     '''Callback to log information about training'''

     def __init__(self):
         self.epoch = 0

     def on_epoch_begin(self, model):
         print("Epoch #{} start".format(self.epoch))

     def on_epoch_end(self, model):
         print("Epoch #{} end".format(self.epoch))
         self.epoch += 1

alldocs = []
with smart_open('data/aclImdb/alldata-id.txt', 'rb', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # 'tags = [tokens[0]]' would also work at extra memory cost
        alldocs.append(TaggedDocument(words=words, tags=tags))
        
class EpochSaver(CallbackAny2Vec):
	'''Callback to save model after each epoch.'''

	def __init__(self, path_prefix):
	 self.path_prefix = path_prefix
	 self.epoch = 0

	def on_epoch_end(self, model):
	 output_path = '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
	 model.save(output_path)
	 self.epoch += 1

print("All docs that will be used for training are: ", len(alldocs))
max_epochs = 100
vec_size = 100
alpha = 0.025

from random import shuffle
doc_list = alldocs[:]  
shuffle(doc_list)

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count() - 1

model = Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0, 
            epochs=20, workers=cores)

model.build_vocab(alldocs)
epoch_logger = EpochLogger()
epoch_saver = EpochSaver(path_prefix="init_model")

model.train(doc_list, total_examples=len(doc_list), epochs=model.epochs, callbacks=[epoch_logger, epoch_saver])

# Save the trained model
