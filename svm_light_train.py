from sklearn import svm, linear_model
import pandas as pd
from tqdm import tqdm
import metrics
import numpy as np
import math
from sklearn.externals import joblib

def load_split(pos_path, neg_path):
	pos = np.loadtxt(pos_path)
	neg = np.loadtxt(neg_path)
	X = np.vstack((pos, neg))
	Y = np.array([1]*pos.shape[0] + [0]*neg.shape[0])

	return X, Y

def train_and_validate(train_paths, val_paths):
	############################## Training Process ###########################################
	X_train, Y_train = load_split(train_paths[0], train_paths[1])
	print(X_train.var())
	print("Started Training")
	clf = svm.SVC(gamma="scale", kernel="rbf", C=1., verbose=False, probability=True)
	#clf = svm.SVC(gamma="scale", kernel="rbf", C=2., verbose=False, probability=True)
	clf.fit(X_train, Y_train)
	joblib.dump(clf, 'data/model_repo/svm_test.pkl')

	# ############################## Validation Process ###########################################
	print("Preparing validation documents")
	X_val, Y_val = load_split(val_paths[0], val_paths[1])

	predictions = clf.predict(X_val)

	np.savetxt("data/errors/system_predictions_bow.txt", clf.predict_proba(X_val))
	np.savetxt("data/errors/gt.txt", Y_val)

	val_accuracy = metrics.acc(predictions, Y_val)
	
	return val_accuracy

fold_val_acc = []
for val_split in range(1,2):
	print("Started iterating Fold: {}".format(val_split))
	pos_train_path = "data/svm_bow/{}/train/pos_train_val_{}_test_0.csv".format(val_split, val_split)
	neg_train_path = "data/svm_bow/{}/train/neg_train_val_{}_test_0.csv".format(val_split, val_split)

	pos_val_path = "data/svm_bow/{}/val/pos_val_val_{}_test_0.csv".format(val_split, val_split)
	neg_val_path = "data/svm_bow/{}/val/neg_val_val_{}_test_0.csv".format(val_split, val_split)

	train_paths = (pos_train_path, neg_train_path)
	val_paths = (pos_val_path, neg_val_path)

	val_accuracy = train_and_validate(train_paths, val_paths)
	print(val_accuracy)
	fold_val_acc.append(val_accuracy)

all_acc = sum(fold_val_acc)/9.

print("The overall accuracy is: ", all_acc)

print("The mean is: ", np.mean(fold_val_acc))
print("The variance is: ", math.sqrt(np.mean(abs(fold_val_acc - np.mean(fold_val_acc))**2)))



