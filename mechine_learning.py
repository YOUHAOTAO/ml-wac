# 导入必要的库
import pandas as pd

# 读取CSV文件，假设文件以逗号分隔，并指定列名称
# 请确保列名称与文件的实际内容相符
df = pd.read_csv('Modified_SQL_Dataset.csv', encoding='utf-8', names=['Query', 'Label'], skiprows=1)

# 查看前几行数据以确认读取是否正确
print(df.head())

# 将数据集分为训练集和测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Query'], df['Label'].astype(int), test_size=0.2, random_state=42)

# 使用TF-IDF特征和朴素贝叶斯分类器创建管道
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率和其他指标
from sklearn.metrics import classification_report, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal', 'Injection'])

# 打印准确率和分类报告
print(f"Accuracy: {accuracy}")
print(report)
