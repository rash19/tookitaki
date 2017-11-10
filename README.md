# tookitaki - Credit card fraud detection

Please run the file in folloing order.
1. Train.py/Train.ipynb
Reads csv file for training data i.e raw_data_70_new,raw_account_70_new and raw_enquiry_70_new. It computes features from training data and save them in a csv file. Please update the filepath to from where you have saved the training files and where you want to save the feature file.

2. Test.py/Test.ipynb
Reads csv file for testing data i.e raw_data_30_new,raw_account_30_new and raw_enquiry_30_new. It computes features from testing data and save them in a csv file. Please update the filepath to from where you have saved the testing files and where you want to save the feature file.

3. Classiifier.py
Reads csv file of feature file for both training and testing data. We have used BalancedBaggingClassifierwith RandomClassifier as native_classifier. It computes the feature ranking and decilde ranking. Please update the filepath to from where you have saved the feature file for training and testing.
