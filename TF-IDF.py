# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import label_binarize
import seaborn as sns

# 读取CSV文件并分配列名称
df = pd.read_csv('Modified_SQL_Dataset.csv', encoding='utf-8', names=['Query', 'Label'], skiprows=1)
df['Label'] = df['Label'].astype(int)  # 确保标签是整型

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['Query'], df['Label'], test_size=0.2, random_state=42)

# 构建管道
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression(random_state=42))
])

# 定义超参数搜索范围
param_grid = {
    'tfidf__max_df': [0.5, 0.7, 0.9],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'lr__C': [0.1, 1.0, 10.0]
}

# 自定义评价函数
scorers = {
    'f1_score': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
}

# 进行超参数搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scorers, refit='f1_score', return_train_score=True, verbose=1)
grid_search.fit(X_train, y_train)

# 使用最佳模型预测测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# 计算F1分数和AUC
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# 打印性能评估
print(f"F1 Score: {f1}")
print(f"AUC Score: {roc_auc}")

# 可视化超参数搜索结果的F1分数和AUC
results_df = pd.DataFrame(grid_search.cv_results_)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(data=results_df, x='param_tfidf__max_df', y='mean_test_f1_score', hue='param_tfidf__ngram_range', style='param_lr__C', ax=ax[0])
sns.lineplot(data=results_df, x='param_tfidf__max_df', y='mean_test_roc_auc', hue='param_tfidf__ngram_range', style='param_lr__C', ax=ax[1])
ax[0].set_title('Hyperparameter Tuning for F1 Score')
ax[0].set_xlabel('TF-IDF Max DF')
ax[0].set_ylabel('Mean Test F1 Score')
ax[1].set_title('Hyperparameter Tuning for AUC')
ax[1].set_xlabel('TF-IDF Max DF')
ax[1].set_ylabel('Mean Test AUC')
plt.show()
