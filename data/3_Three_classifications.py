import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


file_path = './raw data.xlsx'
data = pd.read_excel(file_path)

'''
'''

start_col = 'M1（executive function）'
end_col = 'StddevUnvoicedSegmentLength'
columns_of_interest = data.loc[:, start_col:end_col].columns


cleaned_data = data.dropna(subset=columns_of_interest)


X = cleaned_data.loc[:, start_col:end_col]

y = cleaned_data['Three classification labels（1=Non-lacunar infarction 2=lacunar cerebral infarction；3=Non-cerebral infarction）'].map({1: 0, 2: 1, 3: 2})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

top_features_idx = [100, 25, 18, 40, 52, 87, 45, 57, 36, 53]
top_features = X_train.columns[top_features_idx]

X_train_selected = X_train.loc[:,top_features]
X_test_selected = X_test.loc[:,top_features]


param_grid = {
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 30],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean','manhattan']
    },
    'dt': {
        'max_depth': [2, 10, 20, 30,None],
        'min_samples_split': [2, 4, 6, 8, 20],
    },
    'lr': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'rf': {
        'n_estimators': [30, 40, 50, 80],
        'max_depth': [15, 20, 25],
        'min_samples_split': [10, 15, 20]
    },
    'xgboost': {
        'n_estimators': [20, 50, 100, 150, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}


classifiers = {
    'knn': KNeighborsClassifier(),
    'lr': LogisticRegression(),
    'rf': RandomForestClassifier(),
    'dt': DecisionTreeClassifier(),
    'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


best_models = {}
for name, classifier in classifiers.items():
    print(f"training {name} classifier...")
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid[name], cv=5, scoring='accuracy', n_jobs= -1)
   
    grid_search.fit(X_train_selected, y_train)
  
    best_models[name] = grid_search.best_estimator_
    print("best parameters:", grid_search.best_params_)
    print("best validation score:", grid_search.best_score_)


results = {}


for name, model in best_models.items():
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    print(f"EVA：{name}")
    print(f"ACC: {accuracy}")
    print(f"PRE: {precision}")
    print(f"REC: {recall}")
    print(f"F1_Score: {f1}")




