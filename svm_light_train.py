from sklearn import svm
import pandas as pd
from tqdm import tqdm
import metrics
import numpy as np

# def prepare_embeddings(row):
# 	features = eval( "%s" % row[1] )
# 	representation = []
# 	for feat in features:
# 		representation.append(feat[1])
# 	return representation

def load_split(pos_path, neg_path):
	pos = np.loadtxt(pos_path)
	neg = np.loadtxt(neg_path)
	X = np.vstack((pos, neg))
	Y = np.array([1]*pos.shape[0] + [0]*neg.shape[0])

	return X, Y

############################## Training Process ###########################################
X_train, Y_train = load_split("data/svm_bow/2/train/pos_train_val_2_test_0.csv", "data/svm_bow/2/train/neg_train_val_2_test_0.csv")

clf = svm.SVC(gamma='scale', verbose=True)
clf.fit(X_train, Y_train)

# ############################## Validation Process ###########################################
# print("Preparing validation documents")
# df_pos_validation_data = pd.read_csv('data/test_svm_bow_data/pos_val_val_1_test_0.csv')
# df_neg_validation_data = pd.read_csv('data/test_svm_bow_data/neg_val_val_1_test_0.csv')

# pos_val = []
# for index, row in tqdm(df_pos_validation_data.iterrows()):
# 	pos_val.append(prepare_embeddings(row))

# pos_val_labels  = [1]*len(pos_val)

# neg_val = []
# for index, row in tqdm(df_neg_validation_data.iterrows()):
# 	neg_val.append(prepare_embeddings(row))

# neg_val_labels  = [0]*len(neg_val)

# validation = pos_val+neg_val
# validation_labels = pos_val_labels + neg_val_labels
# print(validation_labels)

# predictions = clf.predict(validation)
# print(metrics.acc(predictions, validation_labels))


# # # print(train)
# # train a model based on the data
# model = svmlight.learn(train, type='classification', verbosity=3)

# del train
# del df_pos_training_data
# del df_neg_training_data

# df_pos_validation_data = pd.read_csv('data/test_svm_bow_data/pos_val_val_1_test_0.csv')
# df_neg_validation_data = pd.read_csv('data/test_svm_bow_data/neg_val_val_1_test_0.csv')

# print("Preparing validation documents")
# pos_val = []
# for index, row in tqdm(df_pos_validation_data[:10].iterrows()):
# 	features = eval( "%s" % row[1] )
# 	train_sample = (0, features)
# 	pos_val.append(train_sample)

# neg_val = []
# for index, row in tqdm(df_neg_validation_data[:10].iterrows()):
# 	features = eval( "%s" % row[1] )
# 	train_sample = (0, features)
# 	neg_val.append(train_sample)

# validation = pos_val + neg_val

# # # classify the test data. this function returns a list of numbers, which represent
# # # the classifications.
# predictions = svmlight.classify(model, validation)
# print(predictions)
# for p in predictions:
#     print '%.8f' % p
