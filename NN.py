from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import metrics, cross_validation
import tensorflow as tf
import pandas as pd
import numpy as np 
from tensorflow.contrib import skflow
from tensorflow.contrib import learn

import xgboost as xgb
from collections import OrderedDict
from sklearn.cross_validation import *
from sklearn.metrics import *
from sklearn.grid_search import GridSearchCV
from pprint import pprint

def clean_my_df(df):
    categorical_data = []
    numerical_data = []

    for i in open("dataDict").read().strip().split("\n") :
    
        if i.split("\t")[2].strip() == "Continuous" :
            numerical_data.append(i.split("\t")[0])
        else :
            categorical_data.append(i.split("\t")[0])

    numeric_features = df[numerical_data]
    categorical_features = df[categorical_data]

    for col_name in categorical_features.columns :
        categorical_features[col_name] = categorical_features[col_name].astype("category",categories=pd.unique(categorical_features[col_name].values.ravel()))
        
        data_oneHot = pd.get_dummies(categorical_features[col_name],prefix =col_name) # One hot encoding
        numeric_features = pd.concat([numeric_features,data_oneHot],axis = 1)

    return numeric_features

train_data = pd.read_excel("Health_care_Dataset_for_probelm.xlsx.xlsx",sheetname="Training Data")

labels = train_data["Lung_Cancer"]
train_data.drop("Lung_Cancer",axis = 1, inplace = True)
train_data.drop("Patient_ID",axis = 1,inplace=True)
print (train_data.shape)
features = clean_my_df(train_data)
print(features.shape)

eval_data = pd.read_excel("Health_care_Dataset_for_probelm.xlsx.xlsx", sheetname="Evaluation Data")
eval_data.drop("Lung_Cancer",axis = 1, inplace = True)
eval_data.drop("Patient_ID",axis = 1,inplace=True)
print(eval_data.shape)
test_features = clean_my_df(eval_data)
print(test_features.shape)


# Check feature importances for the classifier.
xgb_params = {"objective": "binary:logistic","max_depth": 8,"silent":1}
num_rounds = 500
xg_train = xgb.DMatrix(features,label=labels)
bst = xgb.train(xgb_params, xg_train, num_rounds)
importances = bst.get_fscore()
importances = OrderedDict(sorted(importances.items(), key=lambda x: x[1]))

# Important Features
imp_features = features[importances.keys()[-40:]]
imp_test_features = test_features[importances.keys()[-40:]]


# x_train, x_test, y_train, y_test = cross_validation.train_test_split(np.array(features),np.array(labels),test_size=0.5,random_state=8)

# print(x_test.shape)

classifier = learn.DNNClassifier(hidden_units=[100],n_classes=2,activation_fn = tf.nn.relu)
classifier.fit(imp_features,labels,steps=2000)
pred = classifier.predict_proba(imp_test_features)
pd.DataFrame(pred).to_csv("result_NN.csv",index=False)
# print (metrics.classification_report(y_test,pred))
# # score = metrics.accuracy_score(y_test,pred)
# # print('Accuracy: {0:f}'.format(score))
