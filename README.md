# SVM classification in Python

Training and Testing a Support Vector Machine model using the LIBSVM Library. https://www.csie.ntu.edu.tw/~cjlin/libsvm/

In this classification example, X inputs are HOG features extracted from images (in another function) and Y targets their also extracted labels.  
The labels are initially strings (a list of strings) and we use the encoder to transform them to int_32.  
The whole dataset (hog_features, Y_train) is then splitted to train and test sets with the train_test_split function.  
Next step is assigning the libsvm svm problem and parameters and training the model with svm_train.  
Finally, the accuracy of the model is checked on the test set with svm_predict and a whole classification report is presented.
 
