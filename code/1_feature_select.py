import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# 加载数据
file_path = './raw data.xlsx'
data = pd.read_excel(file_path)

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
y = y.map({1: 0, 2: 1})

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用支持向量机（SVC）作为RFECV的基础模型
svc = SVC(kernel="linear")

# 使用RFECV进行特征选择
rfecv = RFECV(estimator=svc, step=1, cv=5)
rfecv.fit(X_train_scaled, y_train)

# 获取特征排名和支持特征
ranking = rfecv.ranking_
supported_features = X.columns[rfecv.support_]

# 提取排名前10的特征
top_10_indices = np.argsort(ranking)[:10]  # 提取排名前10的特征索引
top_10_features = X.columns[top_10_indices]  # 对应的特征名称

# 获取这些特征的原始数据
top_10_data = X[top_10_features]

# 计算前10特征的相关性矩阵
correlation_matrix = top_10_data.corr()

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, square=True,
            annot_kws={"weight": "bold"},
            cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Heatmap of Top 10 Features", fontsize=16, weight="bold")
plt.xticks(weight="bold")
plt.yticks(weight="bold")
plt.tight_layout()
plt.savefig('heatmap_correlation.jpg', dpi=500)
plt.show()

