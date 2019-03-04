# -*- coding: utf-8 -*-

# %% import libraries

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
%matplotlib qt5

# %% read data and explore

df = pd.read_csv("mushrooms.csv", sep=",")

print("Number of missing values:", len(df[df.isnull().any(axis=1)])) # no missing values!

# %% example rows

df.head()
list(df)

# %% describe data

pd.set_option('display.max_columns', 25) # to see all features
df.describe()

# %% detail of veil-type

df['veil-type'].unique()

# %% remove constant-valued veil-type

df = df.drop(columns=['veil-type'])
pd.set_option('display.max_columns', 0) # change max_columns back

# %% encoding to be able to run ML algorithms

le = LabelEncoder()

df = df.apply(le.fit_transform)

# %% histogram of data

df.hist();

# %% correlation matrix

corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True);

# %% check if dataset is balanced

df['class'].unique()
len(df[df['class']==1])/len(df)

# %% name the label and features

label = df['class']
features = df.drop(columns=['class'])

# %% train/Test Split

features_train, features_test, label_train, label_test = train_test_split(features,label, test_size=0.25, random_state = 42, shuffle=True, stratify=label)

features_train, features_valid, label_train, label_valid = train_test_split(features_train,label_train, test_size=(1/3), random_state = 42, shuffle=True, stratify=label_train)

# %% feature importances using RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
rf.fit(features_train, label_train)

feature_names = list(features_train)

print ("Features sorted by their score:\n")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_names), reverse=True))

# %% remove features with low importance
        
df = df.drop(columns=['veil-color'])
df = df.drop(columns=['gill-attachment'])

# %% name the label and features

label = df['class']
features = df.drop(columns=['class'])

# %% train/Test Split

features_train, features_test, label_train, label_test = train_test_split(features,label, test_size=0.25, random_state = 42, shuffle=True, stratify=label)

features_train, features_valid, label_train, label_valid = train_test_split(features_train,label_train, test_size=(1/3), random_state = 42, shuffle=True, stratify=label_train)

# %% Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(features_train, label_train)
label_train_pred = gnb.predict(features_train)

print ("\nGaussian NB Performance on Training Data:")
print ("Accuracy: {0:.3f}".format(accuracy_score(label_train, label_train_pred)))
print ("Recall: {0:.3f}".format(recall_score(label_train, label_train_pred)))
print ("Precision: {0:.3f}".format(precision_score(label_train, label_train_pred)))
print ("f1: {0:.3f}".format(f1_score(label_train, label_train_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_train, label_train_pred)))

label_valid_pred = gnb.predict(features_valid)

print ("\nGaussian NB Performance on Validation Data:")
print ("Accuracy: {0:.3f}".format(accuracy_score(label_valid, label_valid_pred)))
print ("Recall: {0:.3f}".format(recall_score(label_valid, label_valid_pred)))
print ("Precision: {0:.3f}".format(precision_score(label_valid, label_valid_pred)))
print ("f1: {0:.3f}".format(f1_score(label_valid, label_valid_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_valid, label_valid_pred)))

# %% Logistic Regression

lr = LogisticRegression(random_state=22)
lr.fit(features_train, label_train)
label_train_pred = lr.predict(features_train)

print ("\nLogistic Regression Performance on Training Data:")
print ("Accuracy: {0:.3f}".format(accuracy_score(label_train, label_train_pred)))
print ("Recall: {0:.3f}".format(recall_score(label_train, label_train_pred)))
print ("Precision: {0:.3f}".format(precision_score(label_train, label_train_pred)))
print ("f1: {0:.3f}".format(f1_score(label_train, label_train_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_train, label_train_pred)))

# %% Logistic Regression - Hyperparameter Tuning

grid={"C":[0.01,0.1,1.0,10.0,100.0,1000.0], "penalty":["l1","l2"]}

lr_cv=GridSearchCV(lr,grid, n_jobs=-1) # use all processors
lr_cv.fit(features_valid,label_valid)

print("Tuned Hyperparameters:",lr_cv.best_params_)

lr = LogisticRegression(random_state=22, C=1000.0, penalty="l2")
lr.fit(features_train, label_train)

label_valid_pred = lr.predict(features_valid)

print ("\nLogistic Regression Performance on Validation Data:")
print ("Accuracy: {0:.3f}".format(accuracy_score(label_valid, label_valid_pred)))
print ("Recall: {0:.3f}".format(recall_score(label_valid, label_valid_pred)))
print ("Precision: {0:.3f}".format(precision_score(label_valid, label_valid_pred)))
print ("f1: {0:.3f}".format(f1_score(label_valid, label_valid_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_valid, label_valid_pred)))

# %% Decision Tree

dt = DecisionTreeClassifier(random_state=22)
dt.fit(features_train, label_train)
label_train_pred = dt.predict(features_train)

print ("\nDecision Tree Performance on Training Data:")
print ("Accuracy: {}".format(accuracy_score(label_train, label_train_pred)))
print ("Recall: {}".format(recall_score(label_train, label_train_pred)))
print ("Precision: {}".format(precision_score(label_train, label_train_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_train, label_train_pred)))

label_valid_pred = dt.predict(features_valid) # No hyperparameter tuning

print ("\nDecision Tree Performance on Validation Data:")
print ("Accuracy: {}".format(accuracy_score(label_valid, label_valid_pred)))
print ("Recall: {}".format(recall_score(label_valid, label_valid_pred)))
print ("Precision: {}".format(precision_score(label_valid, label_valid_pred)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_valid, label_valid_pred)))

# %% Final Performances

label_test_pred_gnb = gnb.predict(features_test)

print ("\nGaussian NB Performance on Test Data:")
print ("Accuracy: {0:.3f}".format(accuracy_score(label_test, label_test_pred_gnb)))
print ("Recall: {0:.3f}".format(recall_score(label_test, label_test_pred_gnb)))
print ("Precision: {0:.3f}".format(precision_score(label_test, label_test_pred_gnb)))
print ("f1: {0:.3f}".format(f1_score(label_test, label_test_pred_gnb)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_test, label_test_pred_gnb)))

label_test_pred_lr = lr.predict(features_test)

print ("\nLogistic Regression Performance on Test Data:")
print ("Accuracy: {0:.3f}".format(accuracy_score(label_test, label_test_pred_lr)))
print ("Recall: {0:.3f}".format(recall_score(label_test, label_test_pred_lr)))
print ("Precision: {0:.3f}".format(precision_score(label_test, label_test_pred_lr)))
print ("f1: {0:.3f}".format(f1_score(label_test, label_test_pred_lr)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_test, label_test_pred_lr)))

label_test_pred_dt = dt.predict(features_test)

print ("\nDecision Tree Performance on Test Data:")
print ("Accuracy: {0:.3f}".format(accuracy_score(label_test, label_test_pred_dt)))
print ("Recall: {0:.3f}".format(recall_score(label_test, label_test_pred_dt)))
print ("Precision: {0:.3f}".format(precision_score(label_test, label_test_pred_dt)))
print ("f1: {0:.3f}".format(f1_score(label_test, label_test_pred_dt)))
print ("Confusion Matrix: \n{}\n".format(confusion_matrix(label_test, label_test_pred_dt)))

# %% Plot ROC Curve

plt.figure()

pred = label_test_pred_gnb
label = label_test
fpr, tpr, thresh = roc_curve(label, pred)
auc = roc_auc_score(label, pred)
plt.plot(fpr,tpr,label="Gaussian Naive Bayes, AUC="+str(auc))

pred = label_test_pred_lr
label = label_test
fpr, tpr, thresh = roc_curve(label, pred)
auc = roc_auc_score(label, pred)
plt.plot(fpr,tpr,label="Logistic Regression, AUC="+str(auc))

pred = label_test_pred_dt
label = label_test
fpr, tpr, thresh = roc_curve(label, pred)
auc = roc_auc_score(label, pred)
plt.plot(fpr,tpr,label="Decision Tree, AUC="+str(auc))

plt.legend(loc="lower right", fontsize='xx-large')

plt.show()
