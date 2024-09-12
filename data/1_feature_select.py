import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
from skrebate import ReliefF

warnings.filterwarnings("ignore")


file_path = './raw data.xlsx'
data = pd.read_excel(file_path)

start_col = 'M1（executive function）'
end_col = 'StddevUnvoicedSegmentLength'
columns_of_interest = data.loc[:, start_col:end_col].columns

cleaned_data = data.dropna(subset=columns_of_interest)


X = cleaned_data.loc[:, start_col:end_col]
y = cleaned_data['Binary classification label']

y = y.map({1: 0, 2: 1})


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
importance_rf = rf.feature_importances_

# ReliefF
relief = ReliefF(n_neighbors=2, n_features_to_select=len(columns_of_interest), discrete_threshold=10)
relief.fit(X_train_scaled, y_train.to_numpy())
importance_relief = relief.feature_importances_

# LASSO
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
importance_lasso = np.abs(lasso.coef_)
sorted_idx_lasso = np.argsort(importance_lasso)[::-1]
top_features_idx_lasso = sorted_idx_lasso[:10]
top_features_lasso = X_train.columns[top_features_idx_lasso]
top_importance_lasso = importance_lasso[top_features_idx_lasso]
top_importance_lasso_normalized = top_importance_lasso / np.sum(top_importance_lasso)


sorted_idx_rf = np.argsort(importance_rf)[::-1]
sorted_idx_relief = np.argsort(importance_relief)[::-1]

top_features_idx_rf = sorted_idx_rf[:10]
top_features_idx_relief = sorted_idx_relief[:10]

top_features_rf = X_train.columns[top_features_idx_rf]
top_features_relief = X_train.columns[top_features_idx_relief]

top_importance_rf = importance_rf[top_features_idx_rf]
top_importance_relief = importance_relief[top_features_idx_relief]


top_importance_rf_normalized = top_importance_rf / np.sum(top_importance_rf)
top_importance_relief_normalized = top_importance_relief / np.sum(top_importance_relief)


average_importance = (top_importance_rf_normalized + top_importance_relief_normalized) / 2


average_sorted_idx = np.argsort(average_importance)[::-1]
top_features_avg = top_features_rf[average_sorted_idx]
top_importance_avg = average_importance[average_sorted_idx]


print("Random Forest Feature Importances:")
for feature, importance in zip(top_features_rf, top_importance_rf_normalized):
    print(f"feature: {feature}, importance: {importance}")

print("\nLASSO Feature Importances:")
for feature, importance in zip(top_features_lasso, top_importance_lasso_normalized):
    print(f"feature: {feature}, importance: {importance}")

print("\nReliefF Feature Importances:")
for feature, importance in zip(top_features_relief, top_importance_relief_normalized):
    print(f"feature: {feature}, importance: {importance}")

print("\nAverage Feature Importances:")
for feature, importance in zip(top_features_avg, top_importance_avg):
    print(f"feature: {feature}, importance: {importance}")


X_train_selected_avg = pd.DataFrame(X_train, columns=columns_of_interest).loc[:, top_features_avg]



plt.rcParams['font.family'] = 'Times New Roman'

corr_matrix_avg = X_train_selected_avg.corr()

plt.figure(figsize=(20, 19))
heatmap = sns.heatmap(corr_matrix_avg, annot=True, fmt=".2f", cmap='coolwarm',
            square=True, annot_kws={"size":25, "weight":"bold"},
            cbar_kws={"ticks": [-1, -0.5, 0, 0.5, 1], "format": '%.1f'})

cbar = heatmap.collections[0].colorbar
cbar.set_ticks([-1,-0.5,0,0.5,1])
cbar.ax.tick_params(labelsize=25, width=2)
for label in cbar.ax.get_yticklabels():
    label.set_fontsize(25)
    label.set_fontweight('bold')

heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=25, weight='bold')
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=25, weight='bold')

plt.tight_layout(rect=[0, 0, 0.9, 1])


plt.title('Average Top 10 Features Correlation Matrix', fontsize=25, weight='bold', pad=20)
plt.savefig('heatmap_avg.jpg')
plt.close()