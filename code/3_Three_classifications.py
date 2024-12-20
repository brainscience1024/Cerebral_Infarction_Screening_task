import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 加载数据
file_path = "./raw data.xlsx"
data = pd.read_excel(file_path)

'''
(Mocab量表的得分或语音特征有缺失的直接删除该样本；另外，“equivalentSoundLevel_dBp”
特征缺失较多，故暂未考虑该特征，以免所剩样本过少。)
'''
# 指定特征列
start_col = 'M1（executive function）'
end_col = 'StddevUnvoicedSegmentLength'
columns_of_interest = data.loc[:, start_col:end_col].columns

# 清洗数据：删除含有缺失值的行
cleaned_data = data.dropna(subset=columns_of_interest)

# 选择特征列和目标列
X = cleaned_data.loc[:, start_col:end_col]
y = cleaned_data['腔隙性分组（1=非腔隙；2=腔隙；3=非脑梗）'].map({1: 0, 2: 1, 3: 2})

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)

top_features_idx = [60,94,99,26,100,97,30,12,11,72]

top_features = X_train.columns[top_features_idx]
# 使用筛选后的特征进行训练
X_train_selected = X_train.loc[:,top_features]
X_test_selected = X_test.loc[:,top_features]

# 定义多个分类器的网格参数
param_grid = {
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 30],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean','manhattan']
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
    'dt': {
        'max_depth': [2, 10, 20, 30, None],
        'min_samples_split': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    'xgboost': { 
        'eval_metric':['logloss', 'merror', 'mlogloss'],
        'n_estimators': [1000, 1100, 1200, 1300, 1500],
        'max_depth': [ 4, 5, 6, 7, None],
        'subsample' : [ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }
}

# 初始化分类器
classifiers = {
    'knn': KNeighborsClassifier(),
    'lr': LogisticRegression(),
    'rf': RandomForestClassifier(),
    'dt': DecisionTreeClassifier(),
    'xgboost': XGBClassifier(learning_rate = 0.001)
}

# 存储最佳模型
best_models = {}
for name, classifier in classifiers.items():
    print(f"training {name} classifier...")
    cv_folds = 10
    random_seed = 42
    cv = StratifiedKFold(n_splits=cv_folds, random_state=random_seed, shuffle=True)
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid[name], cv=cv, scoring='accuracy', n_jobs=-1)    
    # 训练模型
    grid_search.fit(X_train_selected, y_train)
    # 访问最佳模型
    best_models[name] = grid_search.best_estimator_
    print("best parameters:", grid_search.best_params_)
    print("best validation score:", grid_search.best_score_)

# 初始化字典
results = {}

# 评估模型
for name, model in best_models.items():
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    y_prob = model.predict_proba(X_test_selected)
    auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC':auc_score
    }
    print(f"评估：{name}")
    print(f"准确率: {accuracy}")
    print(f"精确度: {precision}")
    print(f"召回率: {recall}")
    print(f"F1分数: {f1}")
    print(f"AUC (多分类): {auc_score}")
    
