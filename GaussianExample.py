import numpy as np

#list of FEATURES
x = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])

#list of LABELS that will try to put x values into
y = np.array([1,1,1,2,2,2])

from sklearn.naive_bayes import GaussianNB

#creating the classifier that will try to put features of x into a label of y
clf = GaussianNB()

#After creation, it's time to fit the classifier with the data
clf.fit(x,y)


#asking the classifier what label it thinks the point belongs to
print(clf.predict([[-0.8,-1]]))

