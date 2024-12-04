import pymysql

import pandas as pd
#用面向对象的方式编写，更加熟悉面向对象代码风格

class Mysql_csv(object):
    #定义一个init方法，用于读取数据库
    def __init__(self):
        #读取数据库和建立游标对象
        self.connect = pymysql.connect(host="127.0.0.1",port=3306,user="root",password="a123456",database="dataset")
        self.cursor = self.connect.cursor()
        print("success")
    #定义一个del类，用于运行完所有程序的时候关闭数据库和游标对象
    def __del__(self):
        self.connect.close()
        self.cursor.close()

    def write_mysql(self):
        directory='C:\\Users\\13522\\Desktop\\code_bishe\\data\\data.csv'
        sql = "INSERT INTO file_directories (directory_path) VALUES (%s)"
        self.cursor.execute(sql, (directory,))
        self.commit()
        print("\n数据植入完成")
    def commit(self):
        #定义一个确认事务运行
        self.connect.commit()
    def create(self):
        #若已有数据表，则删除
        query="drop table if exists file_directories;"
        self.cursor.execute(query)

        sql = "CREATE TABLE IF NOT EXISTS file_directories (id INT AUTO_INCREMENT PRIMARY KEY,directory_path VARCHAR(255))"
        self.cursor.execute(sql)
        self.commit()
    #运行程序，记得要先调用创建数据的类，在创建写入数据的类
    def run(self):
        self.create()
        self.write_mysql()

#最后用一个main()函数来封装
def main():
    sql = Mysql_csv()
    sql.run()
if __name__ == '__main__':
    main()