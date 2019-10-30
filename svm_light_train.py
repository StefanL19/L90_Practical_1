import svmlight
import preprocess_data_svm

train_data = preprocess_data_svm.train0
test_data = preprocess_data_svm.test0

# train a model based on the data
model = svmlight.learn(train_data, type='classification', verbosity=0)

# classify the test data. this function returns a list of numbers, which represent
# the classifications.
predictions = svmlight.classify(model, test_data)
for p in predictions:
    print '%.8f' % p
