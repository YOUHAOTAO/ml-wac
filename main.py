import sqlite3

# 连接到SQLite数据库
# 数据库文件是test.db
# 如果文件不存在，会自动在当前目录创建:
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# 创建一个表:
cursor.execute('CREATE TABLE user (id varchar(20) primary key, name varchar(20))')

# 插入一行记录，注意使用参数替代:
user_id = '1'
user_name = 'Michael'
cursor.execute('INSERT INTO user (id, name) VALUES (?, ?)', (user_id, user_name))

# 通过参数传递/防注入
user_input = '1 or 1=1'
cursor.execute('SELECT * FROM user WHERE id=?', (user_input,))

print(cursor.fetchall())

# 关闭Cursor和Connection:
cursor.close()
conn.commit()
conn.close()
