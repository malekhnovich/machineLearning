#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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




#########################################################
### your code goes here ###
from sklearn import tree

#create a classifier
#setting min_samples_split to 40 as requested
clf = tree.DecisionTreeClassifier(min_samples_split=40)
#fit the classifier
clf.fit(features_train,labels_train)
#making a prediction


featuresLength  =len(features_train[0])
print(featuresLength)
predict  =clf.predict(features_test)

#judgint the prediction

from sklearn.metrics import accuracy_score

print("The accuracy of the decision tree is "+str(accuracy_score(predict,labels_test)))





#########################################################


