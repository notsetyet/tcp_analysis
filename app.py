import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, session, make_response, url_for
import pickle
from pathlib import Path
import pymysql
import csv
import torch

import pandas as pd

from model.local import train_local, test_local, plot_local, load_and_preprocess_data, get_LSTMModel, load_local_data
from model.remote import train_remote, test_remote, plot_remote, load_remote_data
from model.dupack import train_dupack, test_dupack, plot_dupack, load_dupack_data
from model.retrans import train_retrans, test_retrans, plot_retrans, load_retrans_data
from model.malf import train_malf, test_malf, plot_malf, load_malf_data
from model.ooo import train_outoforder, test_ooo, plot_ooo, load_ooo_data

import sys , os
print(sys.path)
 
app = Flask(__name__)
# 配置MySQL连接（请根据实际情况修改配置信息）
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'a123456'
app.config['MYSQL_DB'] = 'dataset'
# model = pickle.load(open('model.pkl','rb'))

app.static_folder = 'img'

# 建立MySQL连接
conn = pymysql.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)

app.secret_key = 'mykey'  # 设置用于加密会话的密钥，实际应用中请更换为复杂的随机字符串

# 模拟用户数据库（实际应用中应该连接真实数据库进行验证）
users = {
    'admin': '123456'
}

accuracy_loc = 0
f1_loc = 0

accuracy_remo = 0
f1_remo = 0

accuracy_dup = 0
f1_dup = 0

accuracy_ret = 0
f1_ret = 0

accuracy_mal = 0
f1_mal = 0

accuracy_ooo = 0
f1_ooo = 0
 
@app.route('/')
def index():
    return redirect('/login')  # 访问根路径时重定向到登录界面


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return jsonify({"redirect": "/upload"})  # 返回包含重定向目标路径的 JSON 数据
        else:
            return render_template('login.html', error='用户名或密码错误')
    return render_template('login.html')


# @app.route('/upload')
# def upload():
#     if session.get('logged_in'):
#         return render_template('upload.html')
#     return redirect('/login')  # 如果未登录，重定向回登录界面

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if session.get('logged_in'):
        if request.method == 'POST':
            file = request.files['csv_file']
            if file and file.filename.endswith('.csv'):
                # 保存文件到服务器临时目录（这里使用当前目录下的'tmp'文件夹，可根据需求调整）
                if not os.path.exists('tmp'):
                    os.makedirs('tmp')
                file_path = os.path.join('tmp', file.filename)
                print(file_path)
                file.save(file_path)

                # 将CSV文件内容插入到MySQL数据库（这里简单示例插入到名为'csv_data'的表
                try:
                    # with open(file_path, 'r', encoding='utf-8') as csvfile:
                    #     reader = csv.reader(csvfile)
                    #     for row in reader:
                    #         csv_content = ','.join(row)  # 将每行数据转换为字符串形式，可根据实际需求调整存储格式
                    #         cursor.execute("INSERT INTO csv_data (csv_content) VALUES (%s)", (csv_content,))
                    mycursor = conn.cursor()  # 创建游标对象
                    sql = "CREATE TABLE IF NOT EXISTS my_file (id INT AUTO_INCREMENT PRIMARY KEY,directory_path VARCHAR(255))"
                    mycursor.execute(sql)
                    mycursor.execute("INSERT INTO my_file (directory_path) VALUES (%s)", (file_path,))
                    conn.commit()
                except Exception as e:
                    print(f"插入数据到数据库出错: {e}")
                finally:
                    mycursor.close()
                    # os.remove(file_path)  # 插入完成后删除临时文件
                
        
        resp = make_response(render_template('upload.html'))
        resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp
        # return render_template('upload.html')
    return redirect('/login')  # 如果未登录，重定向回登录界面

@app.route('/train', methods=['GET', 'POST'])
def train():
    with conn.cursor() as mycursor:
        # 编写SQL查询语句，这里按照id倒序排列取第一条数据，获取file_path列的值
        # 你可以根据实际表中用于标识顺序的字段（如时间戳字段等）来修改ORDER BY子句
        sql = "SELECT directory_path FROM my_file ORDER BY id DESC LIMIT 1"
        mycursor.execute(sql)
        result = mycursor.fetchone()
        if result:
            data_file_path = result[0]
            
        else:
            print("未查询到相应的数据")
    # window_size = 16
    # step_size = 2
    train_ratio = 0.8
    valid_ratio = 0.1
    num_epochs = 100
    batch_size = 1024
    learning_rate = 0.001
    # model_save_path = '../pickle/model.pth'  # 假设保存模型的路径，可根据实际调整
    img_save_path = './img/'

    global accuracy_loc
    global f1_loc
    global accuracy_remo
    global f1_remo

    global accuracy_dup
    global f1_dup

    global accuracy_ret
    global f1_ret

    global accuracy_mal
    global f1_mal

    global accuracy_ooo
    global f1_ooo

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = load_and_preprocess_data(data_file_path)

    train_dataset, valid_dataset, test_dataset, input_dim, output_dim = load_local_data(df,
                                                                                                train_ratio,
                                                                                                valid_ratio)
    train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1 = train_local(train_dataset,
                                                                                                    valid_dataset,
                                                                                                    input_dim,
                                                                                                    output_dim,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    learning_rate)

    train_file = img_save_path + 'local.png'
    plot_local(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1,
                train_file)
    model = get_LSTMModel(input_dim, output_dim)
    test_file = img_save_path + './cm_local.png'
    accuracy_loc, f1_loc = test_local(model, test_dataset, batch_size, device, test_file)
    
    train_dataset, valid_dataset, test_dataset, input_dim, output_dim = load_remote_data(df,
                                                                                                train_ratio,
                                                                                                valid_ratio)
    train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1 = train_remote(train_dataset,
                                                                                                    valid_dataset,
                                                                                                    input_dim,
                                                                                                    output_dim,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    learning_rate)

    train_file = img_save_path + 'remote.png'
    plot_remote(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1,
                train_file)
    test_file = img_save_path + './cm_remote.png'
    accuracy_remo, f1_remo = test_remote(model, test_dataset, batch_size, device, test_file)

    train_dataset, valid_dataset, test_dataset, input_dim, output_dim = load_dupack_data(df,
                                                                                                train_ratio,
                                                                                                valid_ratio)
    train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1 = train_dupack(train_dataset,
                                                                                                    valid_dataset,
                                                                                                    input_dim,
                                                                                                    output_dim,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    learning_rate)

    train_file = img_save_path + 'dupack.png'
    plot_dupack(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1,
                train_file)
    test_file = img_save_path + './cm_dupack.png'
    accuracy_dup, f1_dup = test_dupack(model, test_dataset, batch_size, device, test_file)
    
    train_dataset, valid_dataset, test_dataset, input_dim, output_dim = load_retrans_data(df,
                                                                                                train_ratio,
                                                                                                valid_ratio)
    train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1 = train_retrans(train_dataset,
                                                                                                    valid_dataset,
                                                                                                    input_dim,
                                                                                                    output_dim,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    learning_rate)

    train_file = img_save_path + 'retrans.png'
    plot_retrans(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1,
                train_file)
    test_file = img_save_path + './cm_retrans.png'
    accuracy_ret, f1_ret = test_retrans(model, test_dataset, batch_size, device, test_file)
    
    train_dataset, valid_dataset, test_dataset, input_dim, output_dim = load_malf_data(df,
                                                                                                train_ratio,
                                                                                                valid_ratio)
    train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1 = train_malf(train_dataset,
                                                                                                    valid_dataset,
                                                                                                    input_dim,
                                                                                                    output_dim,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    learning_rate)

    train_file = img_save_path + 'malf.png'
    plot_malf(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1,
                train_file)
    test_file = img_save_path + './cm_malf.png'
    accuracy_mal, f1_mal = test_malf(model, test_dataset, batch_size, device, test_file)
    
    train_dataset, valid_dataset, test_dataset, input_dim, output_dim = load_ooo_data(df,
                                                                                                train_ratio,
                                                                                                valid_ratio)
    train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1 = train_outoforder(train_dataset,
                                                                                                    valid_dataset,
                                                                                                    input_dim,
                                                                                                    output_dim,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    learning_rate)

    train_file = img_save_path + 'ooo.png'
    plot_ooo(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1,
                train_file)
    
    test_file = img_save_path + './cm_ooo.png'
    accuracy_ooo, f1_ooo = test_ooo(model, test_dataset, batch_size, device, test_file)
    
   
    session["trained"] = True
    return render_template('train.html')  # 如果未登录，重定向回登录界面
    # return render_template('train.html', error='训练还未完成')


@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    session['trained'] = False
    return redirect(url_for('/train'))


@app.route('/test', methods=['GET', 'POST'])
def test():
    if session['trained']:
        return render_template('test.html',  
                               accuracy_loc=accuracy_loc, 
                               f1_loc=f1_loc, 
                               accuracy_remo=accuracy_remo, 
                               f1_remo=f1_remo, 
                               accuracy_dup=accuracy_dup, 
                               f1_dup=f1_dup, 
                               accuracy_ret=accuracy_ret, 
                               f1_ret=f1_ret, 
                               accuracy_mal=accuracy_mal, 
                               f1_mal=f1_mal, 
                               accuracy_ooo=accuracy_ooo, 
                               f1_ooo=f1_ooo)
    return redirect('/train')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80,debug = True)