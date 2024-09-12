import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


file_path = './raw data.xlsx'
data = pd.read_excel(file_path)

start_col = 'M1(executive function）'
end_col = 'StddevUnvoicedSegmentLength'
columns_of_interest = data.loc[:, start_col:end_col].columns


cleaned_data = data.dropna(subset=columns_of_interest)

X = cleaned_data.loc[:, start_col:end_col]
y = cleaned_data['Three classification labels（1=Non-lacunar infarction 2=lacunar cerebral infarction；3=Non-cerebral infarction）'].map({1: 0, 2: 1, 3: 2})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators':[50,100,200],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1,0.2]
}

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')


grid_search.fit(X_train,y_train)


best_model = grid_search.best_estimator_

print("best parameters:", grid_search.best_params_)
print("best validation score:", grid_search.best_score_)


y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"ACC: {accuracy}")
print(f"PRE: {precision}")
print(f"REC: {recall}")
print(f"F1_score: {f1}")


