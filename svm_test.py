import svmlight
import svm_test_data 

training_data = __import__('svm_test_data').train0
test_data = __import__('svm_test_data').test0

# train a model based on the data
model = svmlight.learn(training_data, type='classification', verbosity=3)

# model data can be stored in the same format SVM-Light uses, for interoperability
# with the binaries.
svmlight.write_model(model, 'my_model.dat')

# classify the test data. this function returns a list of numbers, which represent
# the classifications.
# predictions = svmlight.classify(model, test_data)
# for p in predictions:
#     print '%.8f' % p
