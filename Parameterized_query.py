import sqlite3

# 创建或连接到内存数据库
conn = sqlite3.connect(':memory:')  # 使用内存数据库，避免文件操作
cursor = conn.cursor()

# 创建一个示例表
cursor.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, age INTEGER)''')

# 准备插入数据的参数化SQL语句
insert_query = '''INSERT INTO users (username, age) VALUES (?, ?)'''

# 正常数据插入
cursor.execute(insert_query, ('Alice', 30))
cursor.execute(insert_query, ('Bob', 25))

# 模拟SQL注入尝试
malicious_input = "'; DROP TABLE users; --"
try:
    cursor.execute(insert_query, (malicious_input, 0))
    injection_result = "Injection attempt failed: Input was treated as a normal string, not as SQL commands."
except sqlite3.Error as e:
    injection_result = "Caught an SQL error: {}".format(e)

# 尝试读取数据
cursor.execute('SELECT * FROM users')
rows = cursor.fetchall()

# 关闭连接
cursor.close()
conn.close()

# 输出结果
print(rows, injection_result)
