#!/usr/bin/python

""" 
	This is the code to predict survival rates on the Titanic.
	The Titanic survival rate is a kaggle competition.

"""
### Import base modules for data processing
import pandas as pd
import numpy as np
import matplotlib as plt
### Import modules for validatoin
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split, cross_val_score
### Import modules for preprocessing and cleaning data
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
### Import classification module
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.grid_search import GridSearchCV

#Split values based on the two CSVs
training = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv")
testing = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv")

data = training
target = data['Survived'] 

features = pd.concat([data[['Fare', 'Age','Pclass']],
                      pd.get_dummies(data['Sex'], prefix='Sex'),
                      pd.get_dummies(data['Pclass'], prefix='Pclass'),
                      pd.get_dummies(data['Embarked'], prefix='Embarked')
                     ],
                     axis=1)

features = features.drop('Sex_male', 1)
features = features.fillna(-1)

imputer = Imputer(strategy='median', missing_values=-1)
scalar = StandardScaler()
#classifier = GradientBoostingClassifier()

#classifier = GradientBoostingClassifier(n_estimators=100, subsample=.8)
classifier = SVC(class_weight='auto')

params = {
	'clf__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
	'clf__degree': [1, 2, 3],  
	'clf__C': [1, 10, 100, 1000, 10000],
	'clf__gamma': [0.0, 0.1, 0.5, 1.0]
	}

#params = {
#    'clf__learning_rate': [0.2, 0.25, 0.3, 0.35, 0.4],
#    'clf__max_features': [0.74, 0.75, 0.76],
#    'clf__max_depth': [3.5, 3.55, 3.6, 3.65],
#}

pipeline = Pipeline([
    #('scl', scalar),
    ('imp', imputer),
    ('clf', classifier),
])

grid_search = GridSearchCV(pipeline, params, cv=5, scoring='roc_auc')
grid_search.fit(features.values, target)
print grid_search.best_score_
print grid_search.best_params_

