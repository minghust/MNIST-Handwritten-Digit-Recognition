#!/usr/bin/python3
#
## Created by Ming Zhang on 2018-07-04
#

import numpy as np
import pickle
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from loader import LoadData


def loadTrainData(image_path, label_path):
	image_train, label_train = LoadData(image_path, label_path)
	X = np.array(image_train)
	y = np.array(label_train)
	return X, y

def loadTestData(image_path, label_path):
	image_test, label_test = LoadData(image_path, label_path)
	test_image = np.array(image_test)
	test_label = np.array(label_test)
	return test_image, test_label

def train(train_image, train_label):
	#---------------- start train -------------
	clf = KNeighborsClassifier(n_neighbors=5,algorithm='auto',n_jobs=10)
	clf.fit(train_image, train_label)
	return clf

def validate(clf, validate_image, validate_label):
	print('----------------------------------------------------')
	# calculate accuracy and confusion-matrix on VALIDATION data
	confidence = clf.score(validate_image, validate_label)
	print('\nKNN Trained Classifier Confidence: ',confidence)
	
	predicted_label = clf.predict(validate_image)
	accuracy = accuracy_score(validate_label, predicted_label)
	print('\n\nAccuracy of Classifier on Validation Image Data: ',accuracy)

	confusionMatrix = confusion_matrix(validate_label, predicted_label)
	print('\nConfusion Matrix: \n',confusionMatrix)
	print('----------------------------------------------------')

def predict(clf, test_image, test_label):
	# calculate accuracy and confusion-matrix on TEST data
	predicted_label = clf.predict(test_image)
	accuracy = accuracy_score(test_label, predicted_label)
	print('\n\nAccuracy of Classifier on Test Images: ',accuracy)
	
	confusionMatrixTest = confusion_matrix(test_label,predicted_label)
	print('\nConfusion Matrix for Test Data: \n',confusionMatrixTest)
	print('----------------------------------------------------')


if __name__ == '__main__':
	##> load train data and test data
	print('start loading data')
	X, y = loadTrainData('./dataset/train-images.idx3-ubyte', './dataset/train-labels.idx1-ubyte')
	test_image, test_label = loadTestData('./dataset/t10k-images.idx3-ubyte', './dataset/t10k-labels.idx1-ubyte')
	print('finish loading data')

	##> split train data into TRAINING DATA and VALIDATION DATA
	train_image, validate_image, train_label, validate_label = model_selection.train_test_split(X,y,test_size=0.1)

	print('\nstart train')
	##> train
	knn_classifier = train(train_image, train_label)
	print('finish train')

	with open('MNIST_KNN.pickle','wb') as f:
		pickle.dump(knn_classifier, f)

	pickle_in = open('MNIST_KNN.pickle','rb')
	knn_classifier = pickle.load(pickle_in)

	print('\nstart validating')
	##> do validation
	validate(knn_classifier, validate_image, validate_label)
	print('finish validating')

	print('\nstart predicting')
	##> do prediction
	predict(knn_classifier, test_image, test_label)
	print('finish predicting')
