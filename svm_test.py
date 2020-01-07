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


def train_all(train_paths, val_paths):
	"""
		Trains an SVM model on both the whole training set
	"""
	############################## Training Process ###########################################
	X_train, Y_train = load_split(train_paths[0], train_paths[1])
	print(Y_train.shape)
	# ############################## Validation Process ###########################################
	print("Preparing validation documents")
	X_val, Y_val = load_split(val_paths[0], val_paths[1])
	print(Y_val.shape)

	X_all = np.vstack((X_train, X_val))
	Y_all = np.hstack((Y_train, Y_val))
	#same fn
	print("Started Training")
	#clf = svm.SVC(gamma='scale', kernel="rbf", C=2., verbose=False, probability=True)
	clf = svm.SVC(gamma='scale', kernel="rbf", C=1., verbose=False)
	clf.fit(X_all, Y_all)
	joblib.dump(clf, 'data/model_repo/svm_test.pkl')
	
	return clf

# trained_model_path = "data/model_repo/svm_test.pkl"

pos_train_path = "data/svm_bow//1/train/pos_train_val_1_test_0.csv"
neg_train_path = "data/svm_bow/1/train/neg_train_val_1_test_0.csv"

pos_val_path = "data/svm_bow/1/val/pos_val_val_1_test_0.csv"
neg_val_path = "data/svm_bow/1/val/neg_val_val_1_test_0.csv"

train_paths = (pos_train_path, neg_train_path)
val_paths = (pos_val_path, neg_val_path)

clf = train_all(train_paths, val_paths)

test_set_pos_path = "data/svm_bow/1/test/pos_test_val_1_test_0.csv"
test_set_neg_path = "data/svm_bow/1/test/neg_test_val_1_test_0.csv"

X_test, Y_test = load_split(test_set_pos_path, test_set_neg_path)
# clf = joblib.load(trained_model_path)

predictions = clf.predict(X_test)
# np.savetxt("data/predictions/svm_rbf_high.txt", clf.predict_proba(X_test))

test_acc = metrics.acc(predictions, Y_test)

print("The Test Accuracy of the system is: ", test_acc)