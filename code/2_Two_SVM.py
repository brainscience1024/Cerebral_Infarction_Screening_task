import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
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

# 重新编码目标变量：将1映射为0，将2映射为1
y = y.map({1: 1, 2: 0})

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)

top_features_idx = [60,94,99,26,100,97,30,12,11,72]
top_features = X_train.columns[top_features_idx]
# 使用筛选后的特征进行训练
X_train_selected = X_train.loc[:,top_features]
X_test_selected = X_test.loc[:,top_features]

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.fit_transform(X_test_selected)

# 定义网格参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale','auto'],
}

# 初始化svm分类器
svm = SVC(probability=True)
# 设置交叉验证的折数
cv_folds = 10
# 创建StratifiedKFold对象
cv = StratifiedKFold(n_splits=cv_folds, random_state=42, shuffle=True)

# 创建GridSearchCV对象
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=cv, scoring='accuracy')
# 训练模型
grid_search.fit(X_train_scaled,y_train)

# 访问最佳模型
best_model = grid_search.best_estimator_

print("best parameters:", grid_search.best_params_)
print("best validation score:", grid_search.best_score_)

# 在测试集上进行预测
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_selected)[:, 1] 

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
y_prob = best_model.predict_proba(X_test_selected)
auc_score = roc_auc_score(y_test, y_prob[:, 1])

print(f"准确率: {accuracy}")
print(f"精确度: {precision}")
print(f"召回率: {recall}")
print(f"F1分数: {f1}")
print(f"AUC: {auc_score}")
