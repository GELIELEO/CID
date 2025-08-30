import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

# 准备数据
features = np.random.rand(100, 20)  # 示例特征数据
labels = np.random.randint(0, 2, 100)  # 示例二分类标签
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 初始化GBT分类器
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
GBT_model = xgb.XGBClassifier()

# 网格搜索
grid_search = GridSearchCV(GBT_model, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# 最佳参数和模型
print(f"Best parameters: {grid_search.best_params_}")
best_GBT_model = grid_search.best_estimator_

# 预测和评估
predictions = best_GBT_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy with best model: {accuracy}")

# 保存和加载模型
joblib.dump(best_GBT_model, 'GBT_model.pkl')
loaded_model = joblib.load('GBT_model.pkl')
predictions = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy with loaded model: {accuracy}")