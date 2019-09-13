#!/usr/bin/env python
# coding: utf-8

# 1. Multi-class and Multi-Label Classification Using Support Vector Machines
# Importing Libraries

import sklearn
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import collections
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering

# To suppress warnings
warnings.filterwarnings('ignore')

# (a) Download the Anuran Calls (MFCCs) Data Set from GIT local repository
get_ipython().system(' git clone https://github.com/devikasathaye/Breast-Cancer-Diagnostic-Dataset-Monte-Carlo-Simulation')

df_all = pd.read_csv('Breast-Cancer-Diagnostic-Dataset-Monte-Carlo-Simulation/wdbc.csv', sep=',', header=None, skiprows=0)
print("Entire dataset")
df_all

X = df_all.drop(columns=[0,1])
X

y_series = df_all[1]
y = pd.DataFrame(y_series) # contains all the labels

### (b) Monte-Carlo Simulation

### i. Supervised learning

def evaluate_model(y_true, y_pred, df):
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy {}%".format(acc*100))
    prec = precision_score(y_true, y_pred, pos_label='M')
    print("Precision",prec)
    recall = recall_score(y_true, y_pred, pos_label='M')
    print("Recall",recall)
    fsc = f1_score(y_true, y_pred, pos_label='M')
    print("F1-score",fsc)
    auc = roc_auc_score(y_true, df)
    print("AUC",auc)
    return acc, prec, recall, fsc, auc

def draw_roc(target, df):
    # ROC and AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, threshold = roc_curve(target, df, pos_label='M')
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC and AUC')
    plt.legend(loc="lower right")
    plt.show()

def draw_confusion_matrix(target, df):
    cm = confusion_matrix(target, df)
    return cm

# i. Supervised learning
# L1-penalized SVM to classify the data. Use 5 fold cross validation to choose the penalty parameter. Use normalized data.

train_accuracy = []
train_precision = []
train_recall = []
train_fscore =[]
train_areaundercurve = []
test_accuracy = []
test_precision = []
test_recall = []
test_fscore =[]
test_areaundercurve = []

print("Supervised Learning\n")

for m in range(30):
    print("\nIteration #", m+1,"\n")

    X_train, X_test, y_train_series, y_test_series = train_test_split(X, y_series, test_size=0.2, random_state=np.random.randint(10000), stratify=y)

    y_train = pd.DataFrame(y_train_series)
    y_test = pd.DataFrame(y_test_series)

    # normalize data
    X_train_norm = pd.DataFrame(preprocessing.normalize(X_train))
    X_test_norm = pd.DataFrame(preprocessing.normalize(X_test))

    # to show split data
    print("Size of X_train",X_train.shape)
    print("Size of y_train",y_train.shape)
    print("Size of X_test",X_test.shape)
    print("Size of y_test",y_test.shape)
    print("Size of X_train_norm",X_train_norm.shape)
    print("Size of X_test_norm",X_test_norm.shape)

    # to show 20% split(of M and B each)
    y_count = collections.Counter(y_series)
    y_train_count = collections.Counter(y_train_series)
    y_test_count = collections.Counter(y_test_series)
    print("\nNumber of samples belonging to M and B classes in")
    print("Entire dataset:",y_count)
    print("Train dataset:",y_train_count)
    print("Test dataset:",y_test_count)

    b1_acc = []
    b1_c = []
    c_range = np.logspace(-5, 5, 10)
    for c in c_range:
        b1_lsvc = LinearSVC(penalty='l1', C=c, dual=False)
        crossval_scores_std = cross_val_score(b1_lsvc,X_train_norm,y_train,cv=5) # using normalized data
        b1_acc.append(crossval_scores_std.mean())
        b1_c.append(c)
    b1_best_c_norm = b1_c[b1_acc.index(max(b1_acc))]

    print("Best C value is", b1_best_c_norm)
    print("Cross-validation accuracy is {}%".format(max(b1_acc)*100))

    b1_lsvc = LinearSVC(penalty='l1', C=b1_best_c_norm, dual=False)
    b1_lsvc.fit(X_train_norm, y_train)

    b1_train_pred = b1_lsvc.predict(X_train_norm)
    b1_test_pred = b1_lsvc.predict(X_test_norm)

    b1_train_df = b1_lsvc.decision_function(X_train_norm)
    b1_test_df = b1_lsvc.decision_function(X_test_norm)

    print("\nEvaluation metrics")
    print("For train data")
    train_acc, train_prec, train_rec, train_fsc, train_auc = evaluate_model(y_train, b1_train_pred, b1_train_df)
    print("\nFor test data")
    test_acc, test_prec, test_rec, test_fsc, test_auc = evaluate_model(y_test, b1_test_pred, b1_test_df)

    train_accuracy.append(train_acc)
    train_precision.append(train_prec)
    train_recall.append(train_rec)
    train_fscore.append(train_fsc)
    train_areaundercurve.append(train_auc)
    test_accuracy.append(test_acc)
    test_precision.append(test_prec)
    test_recall.append(test_rec)
    test_fscore.append(test_fsc)
    test_areaundercurve.append(test_auc)

### Plot the ROC and report the confusion matrix for training and testing in one of the runs.

print("For M=", m+1, "\n")
# roc for (b)i
print("Train ROC\n")
draw_roc(y_train,b1_train_df)
print("Test ROC\n")
draw_roc(y_test,b1_test_df)
# confusion matrix for (b)i
print("Train Confusion Matrix\n")
tr_cm = draw_confusion_matrix(y_train, b1_train_pred)
print(tr_cm)
print("\nTest Confusion Matrix\n")
te_cm = draw_confusion_matrix(y_test, b1_test_pred)
print(te_cm)

# Report the average accuracy, precision, recall, F-score, and AUC, for both training and test sets over your M runs.
b1_avg_acc_train = np.average(train_accuracy)
b1_avg_prec_train = np.average(train_precision)
b1_avg_recall_train = np.average(train_recall)
b1_avg_fsc_train = np.average(train_fscore)
b1_avg_auc_train = np.average(train_areaundercurve)
b1_avg_acc_test = np.average(test_accuracy)
b1_avg_prec_test = np.average(test_precision)
b1_avg_recall_test = np.average(test_recall)
b1_avg_fsc_test = np.average(test_fscore)
b1_avg_auc_test = np.average(test_areaundercurve)

### ii. Semi-Supervised Learning/ Self-training

# L1-penalized SVM to classify the data. Use 5 fold cross validation to choose the penalty parameter. Use normalized data.

train_accuracy = []
train_precision = []
train_recall = []
train_fscore =[]
train_areaundercurve = []
test_accuracy = []
test_precision = []
test_recall = []
test_fscore =[]
test_areaundercurve = []

print("Semi-Supervised Learning/ Self-training\n")

for m in range(30):
    print("\nIteration #", m+1,"\n")

    X_train, X_test, y_train_series, y_test_series = train_test_split(X, y_series, test_size=0.2, random_state=np.random.randint(10000), stratify=y)
    y_train = pd.DataFrame(y_train_series)
    y_test = pd.DataFrame(y_test_series)
    X_lbl, X_unlbl, y_lbl, y_unlbl = train_test_split(X_train, y_train, test_size=0.5, random_state=np.random.randint(10000), stratify=y_train)

    df_lbl = pd.concat([X_lbl, y_lbl],axis=1)
    df_unlbl = X_unlbl

    # normalize data
    X_lbl_norm = pd.DataFrame(preprocessing.normalize(X_lbl))
    X_unlbl_norm = pd.DataFrame(preprocessing.normalize(X_unlbl))
    X_test_norm = pd.DataFrame(preprocessing.normalize(X_test))

    # to show split data
    print("Size of X_train",X_train.shape)
    print("Size of y_train",y_train.shape)
    print("Size of X_test",X_test.shape)
    print("Size of y_test",y_test.shape)
    print("Size of X_train_norm",X_train_norm.shape)
    print("Size of X_test_norm",X_test_norm.shape)
    print("Size of labelled X_train",X_lbl.shape)
    print("Size of unlabelled X_train",X_unlbl.shape)
    print("Size of labelled X_lbl_norm",X_lbl_norm.shape)
    print("Size of unlabelled X_unlbl_norm",X_unlbl_norm.shape)

    # to show 50% split(of M and B each)
    y_count = collections.Counter(y_series)
    y_lbl_count = collections.Counter(y_train_series)
    print("\nNumber of samples belonging to M and B classes in")
    print("Entire train dataset:",y_count)
    print("Train labelled dataset:",y_lbl_count)
    print("")
    b2a_acc = []
    b2a_c=[]
    c_range = np.logspace(-5, 5, 10)
    for c in c_range:
        b2a_lsvc = LinearSVC(penalty='l1', C=c, dual=False)
        crossval_scores_std = cross_val_score(b2a_lsvc,X_lbl_norm,y_lbl,cv=5) # using normalized data
        b2a_acc.append(crossval_scores_std.mean())
        b2a_c.append(c)
    b2a_best_c_norm = b2a_c[b2a_acc.index(max(b2a_acc))]

    print("Best C value is", b2a_best_c_norm)
    print("Cross-validation accuracy is {}%".format(max(b2a_acc)*100))

    # (b) ii. B
    b2b_lsvc = LinearSVC(penalty='l1', C=b2a_best_c_norm, dual=False)
    while(X_unlbl_norm.shape[0]!=0):
        b2b_lsvc.fit(X_lbl_norm, y_lbl)

        # calculate distances of all unlabelled data points from decision boundary
        distance_to_decision_boundary = b2b_lsvc.decision_function(X_unlbl_norm)
        # calculate index of the farthest point
        farthest_index = distance_to_decision_boundary.tolist().index(max(distance_to_decision_boundary.min(), distance_to_decision_boundary.max(), key=abs))

        # predict label for that point
        data_point = pd.DataFrame(X_unlbl_norm.iloc[[farthest_index]])
        b2b_test_pred = pd.DataFrame(b2b_lsvc.predict(data_point), columns=[1])

        X_lbl_norm = X_lbl_norm.append(data_point)
        y_lbl = y_lbl.append(b2b_test_pred)
        X_unlbl_norm.drop(X_unlbl_norm.iloc[[farthest_index]].index,axis=0,inplace =True)

    print("Test accuracy is {}%".format(b2b_lsvc.score(X_test_norm, y_test)*100))

    b2b_train_pred = b2b_lsvc.predict(X_lbl_norm)
    b2b_test_pred = b2b_lsvc.predict(X_test_norm)

    b2b_train_df = b2b_lsvc.decision_function(X_lbl_norm)
    b2b_test_df = b2b_lsvc.decision_function(X_test_norm)

    print("\nEvaluation metrics")
    print("For train data")
    train_acc, train_prec, train_rec, train_fsc, train_auc = evaluate_model(y_lbl, b2b_train_pred, b2b_train_df)
    print("\nFor test data")
    test_acc, test_prec, test_rec, test_fsc, test_auc = evaluate_model(y_test, b2b_test_pred, b2b_test_df)

    train_accuracy.append(train_acc)
    train_precision.append(train_prec)
    train_recall.append(train_rec)
    train_fscore.append(train_fsc)
    train_areaundercurve.append(train_auc)
    test_accuracy.append(test_acc)
    test_precision.append(test_prec)
    test_recall.append(test_rec)
    test_fscore.append(test_fsc)
    test_areaundercurve.append(test_auc)

### Plot the ROC and report the confusion matrix for training and testing in one of the runs.

print("For M=", m+1, "\n")
# roc for (b)ii
print("Train ROC\n")
draw_roc(y_lbl,b2b_train_df)
print("Test ROC\n")
draw_roc(y_test,b2b_test_df)
# confusion matrix for (b)ii
print("Train Confusion Matrix\n")
tr_cm = draw_confusion_matrix(y_train, b2b_train_pred)
print(tr_cm)
print("\nTest Confusion Matrix\n")
te_cm = draw_confusion_matrix(y_test, b2b_test_pred)
print(te_cm)

# Report the average accuracy, precision, recall, F-score, and AUC, for both training and test sets over your M runs.
b2_avg_acc_train = np.average(train_accuracy)
b2_avg_prec_train = np.average(train_precision)
b2_avg_recall_train = np.average(train_recall)
b2_avg_fsc_train = np.average(train_fscore)
b2_avg_auc_train = np.average(train_areaundercurve)
b2_avg_acc_test = np.average(test_accuracy)
b2_avg_prec_test = np.average(test_precision)
b2_avg_recall_test = np.average(test_recall)
b2_avg_fsc_test = np.average(test_fscore)
b2_avg_auc_test = np.average(test_areaundercurve)

### iii. Unsupervised Learning

def calcdf(alldistances, clus, i):
    num = alldistances[i][1-clus]
    den = np.sum(alldistances[i])
    df = float(num)/float(den)
    return 1 - df

# iii. Unsupervised learning
# Run k-means algorithm on the whole training set. Ignore the labels of the data, and assume k = 2
train_accuracy = []
train_precision = []
train_recall = []
train_fscore =[]
train_areaundercurve = []
test_accuracy = []
test_precision = []
test_recall = []
test_fscore =[]
test_areaundercurve = []

print("Unsupervised Learning\n")

for m in range(30):
    print("\nIteration #", m+1,"\n")

    X_train, X_test, y_train_series, y_test_series = train_test_split(X, y_series, test_size=0.2, random_state=np.random.randint(10000), stratify=y)

    y_train = pd.DataFrame(y_train_series)
    y_test = pd.DataFrame(y_test_series)

    # to show split data
    print("Size of X_train",X_train.shape)
    print("Size of y_train",y_train.shape)
    print("Size of X_test",X_test.shape)
    print("Size of y_test",y_test.shape)

    # to show 20% split(of M and B each)
    y_count = collections.Counter(y_series)
    y_train_count = collections.Counter(y_train_series)
    y_test_count = collections.Counter(y_test_series)
    print("\nNumber of samples belonging to M and B classes in")
    print("Entire dataset:",y_count)
    print("Train dataset:",y_train_count)
    print("Test dataset:",y_test_count)

    kmeans = KMeans(n_clusters=2, random_state=np.random.randint(1000)).fit(X_train)
    # To ensure that the algorithm is not trapped at local minimum,
    # we can choose k observations (rows) at random from data for the initial centroids by passing init=random, n_init=<some integer>,
    # or random_state=np.random.randint(<some integer>)

    train_predictions = kmeans.predict(X_train)
    y_train_copy = y_train.copy()
    y_train_copy['cluster'] = train_predictions

    # calculate distance of each point from each cluster center
    alldistances = kmeans.transform(X_train)
    alldistanceste = kmeans.transform(X_test)

    # calculate distance of each point from it's cluster center
    dist = []
    for i in range(len(train_predictions)):
        dist.append(alldistances[i][train_predictions[i]])

    y_train_copy["distance"] = dist

    # find nearest 30 data points to each cluster
    cls = []
    for cluster in range(2):
        temp_cls = y_train_copy.loc[y_train_copy['cluster']==cluster].nsmallest(30, "distance")[1].to_list()
        cls.append(max(collections.Counter(temp_cls), key=collections.Counter(temp_cls).get))
    print("Clusters:",cls)

    y_train_pred = []
    for i in y_train_copy['cluster']:
        y_train_pred.append(cls[i])

    test_predictions = kmeans.predict(X_test)

    y_test_pred = []
    for i in test_predictions:
        y_test_pred.append(cls[i])

    centroids = kmeans.cluster_centers_
    print("The centers of two clusters are:")
    print(centroids)
    print("")

    b3_train_df = []
    j=0
    for i in train_predictions:
        b3_train_df.append(calcdf(alldistances, i, j))
        j = j+1

    b3_test_df = []
    j=0
    for i in test_predictions:
        b3_test_df.append(calcdf(alldistanceste, i, j))
        j = j+1

    print("\nEvaluation metrics")
    print("For train data")
    train_acc, train_prec, train_rec, train_fsc, train_auc = evaluate_model(y_train, y_train_pred, b3_train_df)
    print("\nFor test data")
    test_acc, test_prec, test_rec, test_fsc, test_auc = evaluate_model(y_test, y_test_pred, b3_test_df)

    train_accuracy.append(train_acc)
    train_precision.append(train_prec)
    train_recall.append(train_rec)
    train_fscore.append(train_fsc)
    train_areaundercurve.append(train_auc)
    test_accuracy.append(test_acc)
    test_precision.append(test_prec)
    test_recall.append(test_rec)
    test_fscore.append(test_fsc)
    test_areaundercurve.append(test_auc)

print("For M=", m+1, "\n")
# roc for (b)i
print("Train ROC\n")
draw_roc(y_train,b3_train_df)
print("Test ROC\n")
draw_roc(y_test,b3_test_df)
# confusion matrix for (b)i
print("Train Confusion Matrix\n")
tr_cm = draw_confusion_matrix(y_train, y_train_pred)
print(tr_cm)
print("\nTest Confusion Matrix\n")
te_cm = draw_confusion_matrix(y_test, y_test_pred)
print(te_cm)

# Report the average accuracy, precision, recall, F-score, and AUC, for both training and test sets over your M runs.
b3_avg_acc_train = np.average(train_accuracy)
b3_avg_prec_train = np.average(train_precision)
b3_avg_recall_train = np.average(train_recall)
b3_avg_fsc_train = np.average(train_fscore)
b3_avg_auc_train = np.average(train_areaundercurve)
b3_avg_acc_test = np.average(test_accuracy)
b3_avg_prec_test = np.average(test_precision)
b3_avg_recall_test = np.average(test_recall)
b3_avg_fsc_test = np.average(test_fscore)
b3_avg_auc_test = np.average(test_areaundercurve)

### iv. Spectral Clustering

# Spectral clustering is a technique with roots in graph theory, where the approach is used to identify communities of nodes in a graph based on the edges connecting them. The method is flexible and allows us to cluster non graph data as well.<br>
# Spectral clustering uses information from the eigenvalues (spectrum) of special matrices built from the graph or the data set. We’ll learn how to construct these matrices, interpret their spectrum, and use the eigenvectors to assign our data to clusters.<br>

def trnsfrm(df,cls):
    dist = pairwise_distances(df, n_jobs=-1, metric="euclidean")
    for i in range(dist.shape[0]):
        dist[i, i] = np.inf
    dist_0 = np.amin(dist[:, np.argwhere(cls == 0).ravel()], axis=1).reshape(-1, 1)
    dist_1 = np.amin(dist[:, np.argwhere(cls == 1).ravel()], axis=1).reshape(-1, 1)
    return np.concatenate((dist_0, dist_1), axis=1)

# iv. Spectral Clustering
# Run k-means algorithm on the whole training set. Ignore the labels of the data, and assume k = 2
train_accuracy = []
train_precision = []
train_recall = []
train_fscore =[]
train_areaundercurve = []
test_accuracy = []
test_precision = []
test_recall = []
test_fscore =[]
test_areaundercurve = []

print("Unsupervised Learning\n")

for m in range(30):
    print("\nIteration #", m+1,"\n")

    X_train, X_test, y_train_series, y_test_series = train_test_split(X, y_series, test_size=0.2, random_state=np.random.randint(10000), stratify=y)

    y_train = pd.DataFrame(y_train_series)
    y_test = pd.DataFrame(y_test_series)

    # normalize data
    X_train_norm = pd.DataFrame(preprocessing.normalize(X_train, axis=0))
    X_test_norm = pd.DataFrame(preprocessing.normalize(X_test, axis=0))

    # to show split data
    print("Size of X_train",X_train.shape)
    print("Size of y_train",y_train.shape)
    print("Size of X_test",X_test.shape)
    print("Size of y_test",y_test.shape)

    # to show 20% split(of M and B each)
    y_count = collections.Counter(y_series)
    y_train_count = collections.Counter(y_train_series)
    y_test_count = collections.Counter(y_test_series)
    print("\nNumber of samples belonging to M and B classes in")
    print("Entire dataset:",y_count)
    print("Train dataset:",y_train_count)
    print("Test dataset:",y_test_count)

    sc = SpectralClustering(n_clusters=2,gamma=1.0, affinity='rbf', random_state=np.random.randint(1000)).fit(X_train_norm)
    # To ensure that the algorithm is not trapped at local minimum,
    # we can choose k observations (rows) at random from data for the initial centroids by passing init=random, n_init=<some integer>,
    # or random_state=np.random.randint(<some integer>)

    train_predictions = sc.labels_
    test_predictions = sc.fit_predict(X_test_norm)
    y_train_copy = y_train.copy()
    y_train_copy['cluster'] = train_predictions

    alldistances = trnsfrm(X_train_norm,train_predictions)
    alldistanceste = trnsfrm(X_test_norm,test_predictions)

    # calculate distance of each point from it's cluster center
    dist = []
    for i in range(len(train_predictions)):
        dist.append(alldistances[i][train_predictions[i]])

    y_train_copy["distance"] = dist

    # find nearest 30 data points to each cluster
    cls = []
    for cluster in range(2):
        temp_cls = y_train_copy.loc[y_train_copy['cluster']==cluster].nsmallest(30, "distance")[1].to_list()
        cls.append(max(collections.Counter(temp_cls), key=collections.Counter(temp_cls).get))
    print("Clusters:",cls)

    y_train_pred = []
    for i in y_train_copy['cluster']:
        y_train_pred.append(cls[i])

    y_test_pred = []
    for i in test_predictions:
        y_test_pred.append(cls[i])

    b4_train_df = []
    j=0
    for i in train_predictions:
        b4_train_df.append(calcdf(alldistances, i, j))
        j = j+1

    b4_test_df = []
    j=0
    for i in test_predictions:
        b4_test_df.append(calcdf(alldistanceste, i, j))
        j = j+1

    print("\nEvaluation metrics")
    print("For train data")
    train_acc, train_prec, train_rec, train_fsc, train_auc = evaluate_model(y_train, y_train_pred, b4_train_df)
    print("\nFor test data")
    test_acc, test_prec, test_rec, test_fsc, test_auc = evaluate_model(y_test, y_test_pred, b4_test_df)

    train_accuracy.append(train_acc)
    train_precision.append(train_prec)
    train_recall.append(train_rec)
    train_fscore.append(train_fsc)
    train_areaundercurve.append(train_auc)
    test_accuracy.append(test_acc)
    test_precision.append(test_prec)
    test_recall.append(test_rec)
    test_fscore.append(test_fsc)
    test_areaundercurve.append(test_auc)

print("For M=", m+1, "\n")
# roc for (b)i
print("Train ROC\n")
draw_roc(y_train,b4_train_df)
print("Test ROC\n")
draw_roc(y_test,b4_test_df)
# confusion matrix for (b)i
print("Train Confusion Matrix\n")
tr_cm = draw_confusion_matrix(y_train, y_train_pred)
print(tr_cm)
print("\nTest Confusion Matrix\n")
te_cm = draw_confusion_matrix(y_test, y_test_pred)
print(te_cm)

# Report the average accuracy, precision, recall, F-score, and AUC, for both training and test sets over your M runs.
b4_avg_acc_train = np.average(train_accuracy)
b4_avg_prec_train = np.average(train_precision)
b4_avg_recall_train = np.average(train_recall)
b4_avg_fsc_train = np.average(train_fscore)
b4_avg_auc_train = np.average(train_areaundercurve)
b4_avg_acc_test = np.average(test_accuracy)
b4_avg_prec_test = np.average(test_precision)
b4_avg_recall_test = np.average(test_recall)
b4_avg_fsc_test = np.average(test_fscore)
b4_avg_auc_test = np.average(test_areaundercurve)

### v. Compare the results you obtained by those methods.

cmptable = []
algo = ['Supervised Learning', 'Semi-supervised Learning', 'Unsupervised Learning', 'Spectral Clustering']
tr_acc = []
tr_pr = []
tr_re = []
tr_fsc = []
tr_auc = []
te_acc = []
te_pr = []
te_re = []
te_fsc = []
te_auc = []

tr_acc.append(b1_avg_acc_train)
tr_acc.append(b2_avg_acc_train)
tr_acc.append(b3_avg_acc_train)
tr_acc.append(b4_avg_acc_train)
tr_pr.append(b1_avg_prec_train)
tr_pr.append(b2_avg_prec_train)
tr_pr.append(b3_avg_prec_train)
tr_pr.append(b4_avg_prec_train)
tr_re.append(b1_avg_recall_train)
tr_re.append(b2_avg_recall_train)
tr_re.append(b3_avg_recall_train)
tr_re.append(b4_avg_recall_train)
tr_fsc.append(b1_avg_fsc_train)
tr_fsc.append(b2_avg_fsc_train)
tr_fsc.append(b3_avg_fsc_train)
tr_fsc.append(b4_avg_fsc_train)
tr_auc.append(b1_avg_auc_train)
tr_auc.append(b2_avg_auc_train)
tr_auc.append(b3_avg_auc_train)
tr_auc.append(b4_avg_auc_train)
te_acc.append(b1_avg_acc_test)
te_acc.append(b2_avg_acc_test)
te_acc.append(b3_avg_acc_test)
te_acc.append(b4_avg_acc_test)
te_pr.append(b1_avg_prec_test)
te_pr.append(b2_avg_prec_test)
te_pr.append(b3_avg_prec_test)
te_pr.append(b4_avg_prec_test)
te_re.append(b1_avg_recall_test)
te_re.append(b2_avg_recall_test)
te_re.append(b3_avg_recall_test)
te_re.append(b4_avg_recall_test)
te_fsc.append(b1_avg_fsc_test)
te_fsc.append(b2_avg_fsc_test)
te_fsc.append(b3_avg_fsc_test)
te_fsc.append(b4_avg_fsc_test)
te_auc.append(b1_avg_auc_test)
te_auc.append(b2_avg_auc_test)
te_auc.append(b3_avg_auc_test)
te_auc.append(b4_avg_auc_test)

cmptable.append(algo)
cmptable.append(tr_acc)
cmptable.append(tr_pr)
cmptable.append(tr_re)
cmptable.append(tr_fsc)
cmptable.append(tr_auc)
cmptable.append(te_acc)
cmptable.append(te_pr)
cmptable.append(te_re)
cmptable.append(te_fsc)
cmptable.append(te_auc)

cmptable=list(map(list,zip(*cmptable)))
cmptable=pd.DataFrame(cmptable, columns=['Algorithm', 'Train Accuracy', 'Train Precision', 'Train Recall', 'Train F1-score', 'Train AUC', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-score', 'Test AUC'])
cmptable

# Semi-supervised Learning seems to have an overall better performance

## 2. Active Learning Using Support Vector Machines

### (a) Download the banknote authentication Data Set from GIT local repository

df_all_2 = pd.read_csv('Breast-Cancer-Diagnostic-Dataset-Monte-Carlo-Simulation/data_banknote_authentication.txt', sep=',', header=None, skiprows=0)
print("Entire dataset")
df_all_2

X = df_all_2.drop(columns=[4])
X

y = pd.DataFrame(df_all_2[4])
y

### (b) Repeat each of the following two procedures 50 times. You will have 50 errors for 90 SVMs per each procedure.

### i. Passive learning

b1_test_err_mc_2 = []
c_range = np.logspace(-5, 5, 10)
for m in range(50):
    print("\nIteration #", m+1,"\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=472, random_state=np.random.randint(1000),stratify = y)

    # to show split data
    print("Size of X_train",X_train.shape)
    print("Size of y_train",y_train.shape)
    print("Size of X_test",X_test.shape)
    print("Size of y_test",y_test.shape)

    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    y_train_copy = y_train.copy()
    y_test_copy = y_test.copy()
    b1_test_err_2 = []
    random_X_train = pd.DataFrame()
    random_y_train = pd.DataFrame()
    while(random_X_train.shape[0]!=900):
        if(random_X_train.shape[0]<500): # randomly picking first 10 samples
            train = pd.concat([X_train_copy, y_train_copy], axis=1)
            rnd_tr = train.sample(n=10,random_state=np.random.randint(1000))

            # to ensure that all samples do not belong to the same class, and there are adequate samples in each class
            while(collections.Counter(rnd_tr[4])[1]!=4):
                rnd_tr = train.sample(n=10,random_state=np.random.randint(1000))
            rnd_X_tr = rnd_tr.drop(columns=[4])
            rnd_y_tr = rnd_tr[4]

            X_train_copy.drop(index=(ind for ind in rnd_X_tr.index),axis=0,inplace =True)
            y_train_copy.drop(index=(ind for ind in rnd_y_tr.index),axis=0,inplace =True)

        else:
            train = pd.concat([X_train_copy, y_train_copy], axis=1)
            rnd_tr = train.sample(n=10,random_state=np.random.randint(1000))
            # to ensure that all samples do not belong to the same class, and there are more than 2 samples in each class
            while(collections.Counter(rnd_tr[4])[1]!=5):
                rnd_tr = train.sample(n=10,random_state=np.random.randint(1000))
            rnd_X_tr = rnd_tr.drop(columns=[4])
            rnd_y_tr = rnd_tr[4]

            X_train_copy.drop(index=(ind for ind in rnd_X_tr.index),axis=0,inplace =True)
            y_train_copy.drop(index=(ind for ind in rnd_y_tr.index),axis=0,inplace =True)

        random_X_train = pd.concat([random_X_train, rnd_X_tr], axis=0) # randomly select <num> data points
        random_y_train = pd.concat([random_y_train, rnd_y_tr], axis=0) # fetch labels of <num> data points

        b1_acc_2 = []
        b1_c_2 = []
        print("Number of samples =",random_X_train.shape[0])
        c = 0.00001
        while c < 100003:
            b1_lsvc_2 = LinearSVC(penalty='l1', C=c, dual=False)
            crossval_scores_std = cross_val_score(b1_lsvc_2, random_X_train, random_y_train, cv=KFold(n_splits=10),n_jobs=-1)# using normalized data
            b1_acc_2.append(crossval_scores_std.mean())
            b1_c_2.append(c)
            c = c * 10
        b1_best_c_2 = b1_c_2[b1_acc_2.index(max(b1_acc_2))]
        print("Best C value is", b1_best_c_2, "with cross-validation accuracy {}%".format(max(b1_acc_2)*100))

        # fit the model using best C
        b1_lsvc_2 = LinearSVC(penalty='l1', C=b1_best_c_2, dual=False)
        b1_lsvc_2.fit(random_X_train, random_y_train)
        b1_test_err_2.append(1-b1_lsvc_2.score(X_test, y_test))
        print("Test error is", b1_test_err_2[len(b1_test_err_2)-1])
        print("")
    print("----------------------------------------------------------------------------")
    b1_test_err_mc_2.append(b1_test_err_2)

### ii. Active Learning

b2_test_err_mc_2 = [] # array for test errors of 50 Monte-Carlo iterations
c_range = np.logspace(-5, 5, 10)
for m in range(50):
    print("\nIteration #", m+1,"\n")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=472, random_state=np.random.randint(1000),stratify = y)

    # to show split data
    print("Size of X_train",X_train.shape)
    print("Size of y_train",y_train.shape)
    print("Size of X_test",X_test.shape)
    print("Size of y_test",y_test.shape)

    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    y_train_copy = y_train.copy()
    y_test_copy = y_test.copy()

    b2_test_err_2 = [] # array for test errors of 90 SVMs
    random_X_train = pd.DataFrame()
    random_y_train = pd.DataFrame()
    while(random_X_train.shape[0]!=900):
        if(random_X_train.shape[0]==0): # randomly picking first 10 samples
            rnd_X_tr = X_train_copy.sample(n=10)
            rnd_y_tr = y_train_copy.loc[y_train_copy.index.isin(rnd_X_tr.index)]

            # to ensure that all samples do not belong to the same class, and there are more than 2 samples in each class
            while(collections.Counter(rnd_y_tr.iloc[0:][4])[0]<=2 or collections.Counter(rnd_y_tr.iloc[0:][4])[0]>=8):
                rnd_X_tr = X_train_copy.sample(n=10)
                rnd_y_tr = y_train_copy.loc[y_train_copy.index.isin(rnd_X_tr.index)]
            X_train_copy.drop(index=(ind for ind in rnd_X_tr.index),inplace =True)
            y_train_copy.drop(index=(ind for ind in rnd_y_tr.index),inplace =True)

        else: # Choose the 10 closest data points in the training set to the hyperplane of the SVM4 and add them to the pool.
            rnd_X_tr = pd.DataFrame()
            rnd_y_tr = pd.DataFrame()
            for z in range(10):
                # calculate distances of all data points from decision boundary
                distance_to_decision_boundary = b2_lsvc_2.decision_function(X_train_copy)
                distance_to_decision_boundary = np.absolute(distance_to_decision_boundary)

                # calculate indices of the 10 nearest points
                ind = distance_to_decision_boundary.tolist().index(min(distance_to_decision_boundary))
                rnd_X_tr = pd.concat([rnd_X_tr,X_train_copy.iloc[[ind]]],axis=0)
                X_train_copy.drop(X_train_copy.iloc[[ind]].index,axis=0,inplace=True)
                rnd_y_tr = pd.concat([rnd_y_tr, y_train_copy.iloc[y_train_copy.index.isin(rnd_X_tr.index)]], axis=0)
                y_train_copy.drop(y_train_copy.iloc[[ind]].index,axis=0,inplace=True)

        random_X_train = pd.concat([random_X_train, rnd_X_tr], axis=0)
        random_y_train = pd.concat([random_y_train, rnd_y_tr], axis=0)

        b2_acc_2 = []
        b2_c_2 = []
        print("Number of samples =",random_X_train.shape[0])
        for c in c_range:
            b2_lsvc_2 = LinearSVC(penalty='l1', C=c, dual=False)
            if(random_X_train.shape[0]==10):
                crossval_scores_std = cross_val_score(b2_lsvc_2, random_X_train, random_y_train, cv=2)
            else:
                crossval_scores_std = cross_val_score(b2_lsvc_2, random_X_train, random_y_train, cv=10)
            b2_acc_2.append(crossval_scores_std.mean())
            b2_c_2.append(c)
        b2_best_c_2 = b2_c_2[b2_acc_2.index(max(b2_acc_2))]
        print("Best C value is", b2_best_c_2, "with cross-validation accuracy {}%".format(max(b2_acc_2)*100))
        b2_lsvc_2 = LinearSVC(penalty='l1', C=b2_best_c_2, dual=False)
        b2_lsvc_2.fit(random_X_train, random_y_train)
        b2_test_err_2.append(1-b2_lsvc_2.score(X_test, y_test))
        print("Test error is", b2_test_err_2[len(b2_test_err_2)-1])
        print("")
    print("-------------------------------------------------------------------------------------")
    b2_test_err_mc_2.append(b2_test_err_2)

### (c) Average the 50 test errors for each of the incrementally trained 90 SVMs in 2(b)i and 2(b)ii.

b1_average_test_error_mc_2 = np.average(b1_test_err_mc_2, axis=0)
b1_average_test_error_mc_2

b2_average_test_error_mc_2 = np.average(b2_test_err_mc_2, axis=0)
b2_average_test_error_mc_2

### Plot average test error versus number of training instances for both active and passive learners on the same figure and report your conclusions. Here, you are actually obtaining a learning curve by Monte-Carlo simulation.

#Learning curve
plt.plot(range(10, 901, 10), b1_average_test_error_mc_2, label='Passive Learning')
plt.plot(range(10, 901, 10), b2_average_test_error_mc_2, label='Active Learning')
plt.xlabel("Number of samples")
plt.ylabel("Test Error Rate")
plt.legend()
plt.show()

# Active Learning initially has high test error, which decreases rapidly.<br>Both types of learning have almost same test errors when number of samples is around 100 plus.
