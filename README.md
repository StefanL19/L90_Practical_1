# L90_Practical
The code for the practical for the L90 course

# Files and organization
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
