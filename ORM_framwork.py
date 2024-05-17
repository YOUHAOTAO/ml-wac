from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 定义模型的基类
Base = declarative_base()

# 定义一个用户模型
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String)
    age = Column(Integer)

# 创建数据库引擎，这里使用内存数据库
engine = create_engine('sqlite:///:memory:')

# 创建表
Base.metadata.create_all(engine)

# 创建会话类
Session = sessionmaker(bind=engine)
session = Session()


# 正常插入数据
user1 = User(username='Alice', age=30)
session.add(user1)

user2 = User(username='Bob', age=25)
session.add(user2)

session.commit()

# 尝试模拟SQL注入
malicious_username = "'; DROP TABLE users; --"
user3 = User(username=malicious_username, age=20)
session.add(user3)
session.commit()

# 查询并打印所有用户数据
users = session.query(User).all()
for user in users:
    print(f'ID: {user.id}, Username: {user.username}, Age: {user.age}')
