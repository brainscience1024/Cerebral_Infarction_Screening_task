import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


file_path = './raw data.xlsx'
data = pd.read_excel(file_path)


start_col = 'M1（executive function）'
end_col = 'StddevUnvoicedSegmentLength'
columns_of_interest = data.loc[:, start_col:end_col].columns


cleaned_data = data.dropna(subset=columns_of_interest)


X = cleaned_data.loc[:, start_col:end_col]
y = cleaned_data['Binary classification label（1=cerebral infarction；2=Non-cerebral infarction）']


y = y.map({1: 0, 2: 1})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

top_features_idx = [ 47, 42, 53, 28, 88, 103, 99, 71, 36, 76]
top_features = X_train.columns[top_features_idx]

X_train_selected = X_train.loc[:,top_features]
X_test_selected = X_test.loc[:,top_features]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.fit_transform(X_test_selected)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale','auto']
}


svm = SVC(probability=True)


grid_search = GridSearchCV(svm, param_grid=param_grid, cv=10, scoring='accuracy')


grid_search.fit(X_train_scaled,y_train)


best_model = grid_search.best_estimator_

print("best parameters:", grid_search.best_params_)
print("best validation score:", grid_search.best_score_)


y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"ACC: {accuracy}")
print(f"PRE: {precision}")
print(f"REC: {recall}")
print(f"F1_score: {f1}")


