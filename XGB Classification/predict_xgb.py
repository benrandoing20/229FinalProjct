import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import pickle
import xgboost as xgb
from sklearn.metrics import confusion_matrix

# Load the saved model from a file
model_file = 'xgboost_model.pkl'  

with open(model_file, 'rb') as file:
    loaded_model = pickle.load(file)

resnet50_features_test_df = pd.read_csv('features/resnet50_feat_test.csv')
features_array_test = resnet50_features_test_df.drop('label', axis=1).values
labels_array_test = resnet50_features_test_df['label'].values

dtest = xgb.DMatrix(features_array_test, label=labels_array_test)

# Make predictions on the test set
y_pred_test = loaded_model.predict(dtest)

# Calculate the confusion matrix
cm = confusion_matrix(y_pred_test, labels_array_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_pred_test, labels_array_test)

print(accuracy)
print(cm)