#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score
import time

clf = svm.SVC(kernel='rbf', C=10000.0)
s = time.time()
print 's', s
clf.fit(features_train, labels_train)
e = time.time()
prediction = clf.predict(features_test)
print accuracy_score(labels_test, prediction)
print 'took', e - s

print sum([1 if p == 1 else 0 for p in prediction])

#########################################################
