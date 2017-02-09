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

def icc(a,b):
    a = np.asarray(a)
    b = np.asarray(b)
    N = a.shape[0]
    if N != b.shape[0]:
        raise 'Shapes do not match'
    x_dash = (a+b).sum() / (2*N)
    s2 = ( ((a-x_dash)**2).sum() + ((b-x_dash)**2).sum() ) / (2*(N-1))
    r = ((a-x_dash)*(b-x_dash)).sum() / ((N-1) * s2)
    return r

def diversion_score(X, offspring_list):
    """
    Offspring list should be a list of tuples (parent_a, offspring, parent_b), where parent_b is optional.
    Result will be in [0,100]. 0 is equivalent to random oversampling, 100 is maximum dissimilarity.
    """
    similarity_sum = 0
    if len(offspring_list[0]) == 2:
        offspring_list = [(parent_a, offspring, parent_a) for (parent_a, offspring) in offspring_list]
    for (parent_a, offspring, parent_b) in offspring_list:
        similarity_sum += max(icc(parent_a, offspring), icc(parent_b, offspring))
    return (1 - (((similarity_sum / len(offspring_list)) + 1) / 2)) * 100 # move from [-1,1] to [0,2], then to [0,1], then inverse, finally move to [0,100]


# check http://stackoverflow.com/questions/23339523/sklearn-cross-validation-with-multiple-scores
