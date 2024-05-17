# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from wordcloud import WordCloud

# 读取CSV文件并分配列名称
df = pd.read_csv('Modified_SQL_Dataset.csv', encoding='utf-8', names=['Query', 'Label'], skiprows=1)

# 转换标签为整数类型
df['Label'] = df['Label'].astype(int)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df['Query'], df['Label'], test_size=0.2, random_state=42)

# 构建管道
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# 定义超参数搜索范围
param_grid = {
    'tfidf__max_df': [0.75, 0.85, 0.95],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'nb__alpha': [0.5, 1.0, 1.5]
}

# 进行超参数搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# 获取最佳参数及其评分
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# 打印最佳参数
print(f"Best Parameters: {best_params}")
print(f"Best Cross-validation Score: {best_score}")

# 使用最佳模型预测测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 打印准确率和分类报告
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal', 'Injection'])

print(f"Test Accuracy: {accuracy}")
print(report)

# 可视化超参数搜索结果
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 6))
sns.lineplot(x='param_tfidf__max_df', y='mean_test_score', hue='param_tfidf__ngram_range', style='param_nb__alpha', data=results)
plt.title('Hyperparameter Tuning Results')
plt.xlabel('TF-IDF Max DF')
plt.ylabel('Mean Test Score')
plt.legend(title='Ngram Range / Alpha')
plt.show()

# 可视化高频词或指令的热力图或词云
# 选择某一类标签的数据，例如SQL注入攻击
sqli_queries = df[df['Label'] == 1]['Query']

# 创建词云
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(sqli_queries))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('SQL Injection Common Words')
plt.show()

