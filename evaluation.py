import sklearn.metrics
import sklearn.linear_model
import sklearn.ensemble
import sklearn.ensemble
import sklearn.svm
import numpy as np
import warnings

import pandas as pd
from IPython.display import display

def evaluate_clustering_methods(methods):
	""" 
	Evaluate the performance of given clusterings 
	Args:
		methods (dictionary) - Should contain dicitionaries describing each method with the properties 'name', 'target', and 'clustering'
	"""
	results = {}
	for m in methods:
	    res = results[m['name']] = {}
	    prec = 3
	    res['Adjusted Rand Score'] = round(sklearn.metrics.adjusted_rand_score(m['target'], m['clustering']),prec)
	    res['Normalized Mutual Information'] = round(sklearn.metrics.normalized_mutual_info_score(m['target'], m['clustering']),prec)
	    res['Adjusted Mutual Information'] = round(sklearn.metrics.adjusted_mutual_info_score(m['target'], m['clustering']),prec)
	return np.transpose(results)

def classify(X, y, X_validate):
	""" Fits different classifiers without parameters """
	# Logistic Regression
	lr = sklearn.linear_model.LogisticRegression()
	# Gradient Boosting Machine
	gb = sklearn.ensemble.GradientBoostingClassifier()
	# Random Forest
	rf = sklearn.ensemble.RandomForestClassifier()
	# Support Vector Machine
	svm = sklearn.svm.SVC()

	classifiers = {
		'Logistic Regression': lr,
		'Random Forest': rf,
		'Gradient Boosting': gb,
		'Support Vector': svm
	}
	predictions = {}
	for name, classifier in classifiers.items():
		classifier.fit(X,y)
		predictions[name] = classifier.predict(X_validate) 
	return predictions

def evaluate_classification(y, y_predicted):
	evaluation = {}
	# convert the labels to multi-label-indicator format (one-hot)
	transformer = sklearn.preprocessing.MultiLabelBinarizer(classes=np.unique(np.concatenate([y,y_predicted])))
	y_multilabel = transformer.fit_transform([[label] for label in y])
	y_predicted_multilabel = transformer.fit_transform([[label] for label in y_predicted])
	# F-Measure
	# If our classifier only predicts majority class, F1 score is set to 0 for minority class.
	# Since we apply an unweighted average of F1 scores of each class, that's ok - it still punishes the classifier.
	with warnings.catch_warnings():
		warnings.filterwarnings(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)
		evaluation['Balanced F-Measure'] = sklearn.metrics.f1_score(y, y_predicted, average='macro')
	# G-Mean
	evaluation['G-Measure'] = g_measure(y, y_predicted)
	# AUC
	try:
		evaluation['ROC AUC Score'] = sklearn.metrics.roc_auc_score(y_multilabel, y_predicted_multilabel, average='macro')
	except ValueError:
		evaluation['ROC AUC Score'] = 0
	return evaluation

def g_measure(y_true,y_pred):
	with warnings.catch_warnings():
		warnings.filterwarnings(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)
    	precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    g_measure = np.sqrt(precision * recall)
    return g_measure

# check http://stackoverflow.com/questions/23339523/sklearn-cross-validation-with-multiple-scores
