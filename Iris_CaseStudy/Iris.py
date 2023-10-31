import numpy as np 
from sklearn import tree 
from sklearn.datasets import load_iris

iris = load_iris()

print("\n\nFeature names of iris data set ")
print(iris.feature_names)

print ("\n\nTarget names of iris data set")
print(iris.target_names)

# Indices of removed elements : 
test_index = [1 , 51 , 101]

# Trainig data with removed elements :
train_target = np.delete(iris.target , test_index)
train_data = np.delete(iris.data , test_index , axis=0)

# Testing data for testing on traing data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

# Form a decision tree classifier 
classifier = tree.DecisionTreeClassifier()

# Apply training data to form tree
classifier.fit(train_data , train_target)

print("\n\n Values that we removed for testing")
print(test_target)

print(" \n\nResults of testing ")
print(classifier.predict(test_data))