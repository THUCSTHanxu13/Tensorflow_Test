import numpy as np
import random
from sklearn import svm, linear_model, datasets, naive_bayes
from init import *

w, x_train, y_train, x_dev, y_dev = load_data_and_labels()

trainX = []
for i in x_train:
	sentence = []
	for j in i:
		sentence.append(w[j])
	res = np.concatenate(sentence, axis = 0)
	trainX.append(res)

trainY = []
for i in y_train:
	s = 0
	for j in i:
		if j == 1:
			trainY.append(s)
			break
		s = s + 1
testY = []
for i in y_dev:
	s = 0
	for j in i:
		if j == 1:
			testY.append(s)
			break
		s = s + 1

testX = []
for i in x_dev:
	sentence = []
	for j in i:
		sentence.append(w[j])
	res = np.concatenate(sentence, axis = 0)
	testX.append(res)


# clf = svm.LinearSVC()
# clf.fit(trainX, trainY)
# print clf.score(testX, testY)

# clf = linear_model.LogisticRegression()
# clf.fit(trainX, trainY)
# print clf.score(testX, testY)

# gnb = naive_bayes.GaussianNB()
# gnb.fit(trainX, trainY)
# print gnb.score(testX, testY)
