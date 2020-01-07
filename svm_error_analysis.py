from sklearn import svm, linear_model
import pandas as pd
from tqdm import tqdm
import metrics
import numpy as np
import math
from sklearn.externals import joblib

predictions_file = "data/errors/system_predictions_bow.txt"
gt_file = "data/errors/gt.txt"
positive_files = "data/svm_doc2vec/1/val/val_docs_pos.txt"
negative_files = "data/svm_doc2vec/1/val/val_docs_neg.txt"

# Load the positive files
with open(positive_files, "r") as pos_f:
	pos_content = pos_f.readlines() 
	pos_content = [x.strip() for x in pos_content]

# Load the negative files
with open(negative_files, "r") as neg_f:
	neg_content = neg_f.readlines() 
	neg_content = [x.strip() for x in neg_content]

preds = np.loadtxt(predictions_file)
gt = np.loadtxt(gt_file)

pos_preds = preds[:100]
neg_preds = preds[100:]

pos_gt = gt[:100]
neg_gt = gt[100:]

pos_problems_path = "data/pos_svm_errors_bow.txt"
neg_problems_path = "data/neg_svm_errors_bow.txt"

#Get Positive problems
for idx, elem in enumerate(pos_gt):
	with open(pos_problems_path, "a") as pf:
		if elem	!= np.argmax(pos_preds, axis=1)[idx]:
			pf.write(pos_content[idx]+"\n")
			# print(pos_content[idx])
			# print("--------------------------------")

#Get Negative Problems
for idx, elem in enumerate(neg_gt):
	with open(neg_problems_path, "a") as nf:
		if elem	!= np.argmax(neg_preds, axis=1)[idx]:
			nf.write(neg_content[idx]+"\n")
			# print(neg_content[idx])
			# print("--------------------------------")
