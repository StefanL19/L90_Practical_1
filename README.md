# L90_Practical
The code for the practical for the L90 course

# Files and organization
### Train a Doc2Vec Model
Use the train_model function in the file train_doc2vec.py

### Prepare Doc2Vec embeddings for SVM
Use the file prepare_doc2vec_data_svm.py

### Prepare BoW features for SVM
Use the file preprocesss_BoW_data_svm.py

### Train an SVM model
Use the file svm_light_train.py and set the appropriate data paths there

### Test an SVM model on the blind test set
Use the file svm_test.py

### Export mistakes for manual error analysis
Use the file svm_error_analysis.py

### Test a Doc2Vec model on the Triplet Accuracy and 'Perfect' Triplet Accuracy Metrics
Use the file test_doc2vec.py

### Permutation Test
Use the file permutation_test.py and pass the appropriate data paths

### Train  a Naive Bayes classifier on cv000-cv899 and test on cv900-cv999
Use the file train_nb_no_fold.py and set the Laplace parameter to False 

### Train a Naive Bayes classifier + Laplace Smoothing on cv000-cv899 and test on cv900-cv999 
Use the file train_nb_no_fold.py and set the Laplace parameter to True

### Train a Naive Bayes classifier on cross-validation
Use the file train_nb_nfold.py

### Sign Tests, Accuracy Tests
Use the function in the file file calculate_metrics.py

### 10-fold Round Robin Cross Validation + Laplace Smoothing
Use the file train_nb_no_fold.py and set the Laplace parameter to True

### 10-fold Round Robin Cross Validation No Smoothing 
Use the file train_nb_no_fold.py and set the Laplace parameter to False

### Generating 10-fold predictions for trained models
Use the file nfold_prediction_generator.py

### Generating holdout set predictions for trained models
Use the file generate_system_predictions.py

### Predictions for some already trained systems
In data/trained_models_new/predictions/

### Pre-trained models on cross-validation
In data/trained_models_new/10_fold_no_test 

WIP
### Train SVM 9-fold Cross Validation + test on the first fold (BoW)
### Train doc2vec on IMDB Dataset
### Train SVM 9-fold Cross Validation + test on the first (docv2vec)
