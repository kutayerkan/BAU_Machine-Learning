# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:05:36 2018

@author: kutay.erkan
"""

"""
References:
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html
https://seaborn.pydata.org/generated/seaborn.lmplot.html
https://cmdlinetips.com/2018/03/pca-example-in-python-with-scikit-learn/
"""

import os
os.getcwd()

# %% Import libraries

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr" # "all" to print multiple statements

# %% Read data

df = pd.read_table("dataset.txt", sep=",", header=None)

# Name features

df.columns = ["word_freq_make", "word_freq_address", "word_freq_all",
              "word_freq_3d", "word_freq_our", "word_freq_over",
              "word_freq_remove", "word_freq_internet", "word_freq_order",
              "word_freq_mail", "word_freq_receive", "word_freq_will",
              "word_freq_people", "word_freq_report", "word_freq_addresses",
              "word_freq_free", "word_freq_business", "word_freq_email",
              "word_freq_you", "word_freq_credit", "word_freq_your",
              "word_freq_font", "word_freq_000", "word_freq_money",
              "word_freq_hp", "word_freq_hpl", "word_freq_george",
              "word_freq_650", "word_freq_lab", "word_freq_labs",
              "word_freq_telnet", "word_freq_857", "word_freq_data",
              "word_freq_415", "word_freq_85", "word_freq_technology",
              "word_freq_1999", "word_freq_parts", "word_freq_pm",
              "word_freq_direct", "word_freq_cs", "word_freq_meeting",
              "word_freq_original", "word_freq_project", "word_freq_re",
              "word_freq_edu", "word_freq_table", "word_freq_conference",
              "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
              "char_freq_$", "char_freq_#", "capital_run_length_average",
              "capital_run_length_longest", "capital_run_length_total",
              "is_spam"
]

# %% Explore data    
    
print (df.head(),"\n")
print ("Number of all instances: {}".format(len(df)))
print ("Number of spam instances: {}\n".format(df.is_spam.sum()))
print ("Features:\n {}".format(list(df)))

# %% Name the label and features

label = df.is_spam
features = df.drop(columns=['is_spam'])

# %% Train/Test Split

features_train, features_test, label_train, label_test = train_test_split(features,label, test_size=0.5, random_state = 22, stratify=df.is_spam)

print ("Number of spam instances in training set: {}".format(label_train.sum()))
print ("Number of spam instances in test set: {}".format(label_test.sum()))

# %% Directly run kNN on training and test data without PCA or feature reduction

clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
clf.fit(features_train, label_train)
label_train_pred = clf.predict(features_train)
label_test_pred = clf.predict(features_test)

print ("\nPerformance on Training Data without Feature Reduction:")
print ("Accuracy: {}".format(accuracy_score(label_train, label_train_pred)))
print ("Recall: {}".format(recall_score(label_train, label_train_pred)))
print ("Precision: {}".format(precision_score(label_train, label_train_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_train, label_train_pred)))

print ("Performance on Test Data without Feature Reduction:")
print ("Accuracy: {}".format(accuracy_score(label_test, label_test_pred)))
print ("Recall: {}".format(recall_score(label_test, label_test_pred)))
print ("Precision: {}".format(precision_score(label_test, label_test_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_test, label_test_pred)))

# %% See accuracy score and cumulative explained variance for different m using training data

selected_n = 0
pca_accuracy = 0

for i in range(1, 58):
    pca = PCA(n_components=i)
    pca.fit(features_train)
    pc = pca.fit_transform(features_train)
    clf.fit(pc,label_train)
    label_train_pred = clf.predict(pc)
    if accuracy_score(label_train, label_train_pred) > pca_accuracy:
        pca_accuracy = accuracy_score(label_train, label_train_pred)
        selected_n = i
    print ("Accuracy for {} features: {}".format(i,accuracy_score(label_train, label_train_pred)))
    print ("Recall for {} features: {}".format(i,recall_score(label_train, label_train_pred)))
    print ("Precision for {} features: {}".format(i,precision_score(label_train, label_train_pred)))
    print ("Cum. explained variance ratio for {} features: {}".format(i,pca.explained_variance_ratio_.cumsum()[-1]))

print ("\nSelected n_components with highest accuracy score: {}".format(selected_n))
print ("Accuracy score for {} components: {}".format(selected_n, pca_accuracy))

# %% Plot for m=2
    
pca = PCA(n_components=2)
pc = pca.fit_transform(features_train)
pc_df = pd.DataFrame(data = pc, columns = ['PC1', 'PC2'])
label_train=label_train.reset_index(drop=True)
pc_df = pd.concat([pc_df,label_train],axis=1)
print ("\nPlot for 2 principal components PC1, PC2:")
sns.lmplot(x="PC1", y="PC2", data=pc_df, hue='is_spam',
           fit_reg=False, scatter_kws={"alpha": 0.2});

# %% See performance metrics for chosen m using training and test data

print('n_components=41 and 42 have the same accuracy. Which one is selected DOES change on runtime.')
print ("\nSelected n_components: {}".format(selected_n))
pca = PCA(n_components=selected_n)
pca.fit(features_train)
pc = pca.fit_transform(features_train)
clf.fit(pc,label_train)
label_train_pred = clf.predict(pc)

print ("\nPerformance on Training Data with Feature Reduction Using PCA:")
print ("Accuracy: {}".format(accuracy_score(label_train, label_train_pred)))
print ("Recall: {}".format(recall_score(label_train, label_train_pred)))
print ("Precision: {}".format(precision_score(label_train, label_train_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_train, label_train_pred)))

pca = PCA(n_components=42)
pca.fit(features_test)
pc = pca.fit_transform(features_test)
clf.fit(pc,label_test)
label_test_pred = clf.predict(pc)

print ("Performance on Test Data with Feature Reduction Using PCA:")
print ("Accuracy: {}".format(accuracy_score(label_test, label_test_pred)))
print ("Recall: {}".format(recall_score(label_test, label_test_pred)))
print ("Precision: {}".format(precision_score(label_test, label_test_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_test, label_test_pred)))
     
# %% Finding feature accuracies

clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn_accuracy = 0

accuracy_list = pd.DataFrame(columns=["feature","accuracy"])

for i in range(len(features_train.columns)):
    
    clf.fit(np.array(features_train.iloc[:,i]).reshape(-1,1), label_train)
    label_train_pred = clf.predict(np.array(features_train.iloc[:,i]).reshape(-1,1))
    knn_accuracy = accuracy_score(label_train, label_train_pred)
    accuracy_temp = pd.DataFrame([[features_train.iloc[:,i].name,knn_accuracy]],columns=["feature","accuracy"])
    accuracy_list = accuracy_list.append(accuracy_temp, ignore_index=True)
    #print ("{} accuracy score: {}".format(features_train.iloc[:,i].name,knn_accuracy))

accuracy_list = accuracy_list.sort_values(by=['accuracy'],ascending=False)
print(accuracy_list)

# %% Finding accuracy, recall, and precision with forward selection

clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

clf.fit(features_train.loc[:,accuracy_list['feature'].iloc[0]].values.reshape(-1,1), label_train)
label_train_pred = clf.predict(features_train.loc[:,accuracy_list['feature'].iloc[0]].values.reshape(-1,1))

print ("Accuracy with 1 features: {}".format(accuracy_score(label_train, label_train_pred)))
print ("Recall with 1 features: {}".format(recall_score(label_train, label_train_pred)))
print ("Precision with 1 features: {}".format(precision_score(label_train, label_train_pred)))

i=2
for i in range(2,58):
    clf.fit(features_train.loc[:,accuracy_list['feature'].iloc[0:i]], label_train)
    label_train_pred = clf.predict(features_train.loc[:,accuracy_list['feature'].iloc[0:i]])
    print ("Accuracy with {} features: {}".format(i,accuracy_score(label_train, label_train_pred)))
    print ("Recall with {} features: {}".format(i,recall_score(label_train, label_train_pred)))
    print ("Precision with {} features: {}".format(i,precision_score(label_train, label_train_pred)))

# %% Plot for 2 features selected by forward selection

select_features = pd.DataFrame(data = features_train, columns = ['char_freq_!', 'char_freq_$'])
label_test=label_test.reset_index(drop=True)
select_features_df = pd.concat([select_features,label_test],axis=1)
print ("\nPlot for 2 principal components char_freq_!, char_freq_$:")
sns.lmplot(x="char_freq_!", y="char_freq_$", data=select_features_df, hue='is_spam',
           fit_reg=False, scatter_kws={"alpha": 1});



















