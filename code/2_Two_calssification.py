import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

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
y = cleaned_data['是否脑梗']

# 重新编码目标变量
y = y.map({1: 1, 2: 0})

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
        'metric': ['euclidean','manhattan']
    },
        'dt': {
        'max_depth': [2, 4, 6, 10, 20],
        'min_samples_split': [12, 10, 8, 6, 4, 2],
        'class_weight': ['balanced']
    },
    'xgboost': {
        'eval_metric':['logloss'],
        'n_estimators': [100, 200, 500, 1000, 2000, 3000, 4000],
        'max_depth': [10, 8, 6, 4, 2, None],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
        'subsample' : [ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    }
}
# 初始化分类器
classifiers = {
    'knn': KNeighborsClassifier(),
    'dt': DecisionTreeClassifier(),
    'xgboost': XGBClassifier()
}

# 存储最佳模型
best_models = {}
for name, classifier in classifiers.items():
    print(f"training {name} classifier...")
    # 使用StratifiedKFold作为交叉验证
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

results = {}

for name, model in best_models.items():
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    y_prob = model.predict_proba(X_test_selected)
    auc_score = roc_auc_score(y_test, y_prob[:, 1])
    results[name]= {
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
    print(f"AUC: {auc_score}")


