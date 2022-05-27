import random

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encoding train labels (strings to integers)
encoder.fit(make)
Y_train = encoder.transform(make)

from sklearn.model_selection import train_test_split

# spliting training and testing sets with random 80/20 split
X_train, X_test, y_train, y_test = train_test_split(hog_features, Y_train, test_size=0.2, random_state=random.randint(1, 100))        

X_train = np.array(X_train,dtype=np.float32)
X_test = np.array(X_test,dtype=np.float32)
y_train = np.array(y_train)
y_test = np.array(y_test)

from svmutil import *
from svm import *

# building the svm_model
prob = svm_problem(y_train, X_train)

param = svm_parameter()
param.kernel_type = LINEAR
param.svm_type = C_SVC
param.probability = 1
model = svm_train(prob, param)

# testing the accuracy of the model
yp_labels, p_acc, y_tests = svm_predict(y_test, X_test, model,"-b 1") #if -b 1, outputs a list of probability estimates is specified
yp_labels = np.array(yp_labels,dtype=int)
#print (yp_labels)
#print (y_test)

# detailed classification report : precision, recall, F1-score!
from sklearn import metrics
print(metrics.classification_report(y_test, yp_labels, digits=3))

#saving the trained model
svm_save_model('svm_model.model', model)

