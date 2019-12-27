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