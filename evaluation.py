import sklearn.metrics
import sklearn.linear_model
import sklearn.ensemble
import sklearn.ensemble
import sklearn.svm

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
	for prediction in y_predicted:
		# F-Measure
		evaluation['Weighted F-Measure'] = sklearn.metrics.f1_score(y, y_predicted, average='weighted')
		# G-Mean
		# AUC
	return evaluation