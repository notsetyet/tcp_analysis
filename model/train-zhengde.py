
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import pandas as pd
# here = Path(__file__).parent
data_file = f'../data/full.csv'
df = pd.read_csv(data_file)
df.columns=['islocal','direct','isserver','duration','framesizeration_in','framesizeratio_out','localdelayavg','localdelaynavg',
            'remotedelayavg','remotedelaynavg','dupack','retrans','malf','outoforder','localdelay_label','remotedelay_label','dupack_label',
           'retrans_label','malf_label','outoforder_label']

print(df.head(10))

df_filled = df.fillna(0)
print(df_filled.columns)


feature_column=df_filled.columns[:14]
# 这里下面对数据的处理能被raw_feature迁移识别吗？
raw_feature=df_filled[feature_column]
localdelay_y=df_filled['localdelay_label']
remotedelay_y=df_filled['remotedelay_label']
dupack_y=df_filled['dupack_label']
retrans_y=df_filled['retrans_label']
malf_y=df_filled['malf_label']
outoforder_y=df_filled['outoforder_label']



from sklearn.preprocessing import StandardScaler
import numpy as np

# 创建一个StandardScaler对象
scaler = StandardScaler()

# 使用fit_transform方法对数据进行标准化
scaled_feature = scaler.fit_transform(raw_feature.iloc[:,[3,4,5,6,7,8,9,10,11,12,13]])

print(f'scaled_feature: {scaled_feature}')

df_scale=df_filled  # 为了不改变原始数据，这里复制一份
print(f'df_scale: {df_scale}')

df_scale.iloc[:,[3,4,5,6,7,8,9,10,11,12,13]]=scaled_feature  # 将标准化后的数据替换原始数据
print(f'df_scale: {df_scale}')  # (2971, 20)


def split_time_series(data, window_size, step_size):
    """
    将时间序列数据切分为较短的窗口。
    
    参数:
        data: 时间序列数据，形状为 (M, N)，M 是时间点数量，N 是特征数量。
        window_size: 每个窗口的大小（时间点数量）。
        step_size: 窗口之间的步长（时间点数量）。
        
    返回:
        切分后的数据，形状为 (num_windows, window_size, N)。
    """
    M, N = data.shape
    num_windows = (M - window_size) // step_size + 1
    windows = np.empty((num_windows, window_size, N-1))
    labels=[]
    
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx,:-1]

        labels.append( data[end_idx-1,-1])
        
    return windows,np.array(labels)

# window_size = 128  # 每个窗口包含20个时间点
# step_size = 32 # 窗口之间的步长为32个时间点
window_size = 16  # 每个窗口包含20个时间点
step_size = 2 # 窗口之间的步长为32个时间点
# localdelay
raw_local = df_scale.iloc[:, :15]
features, labels = split_time_series(raw_local.values, window_size, step_size)  # 切分数据

cnn_lstm_att_features=torch.tensor(features, dtype=torch.float)  # (89, 128, 18)
cnn_lstm_att_labels=torch.tensor(labels, dtype=torch.float)  # (89)


print(f"cnn_lstm_att_features shape: {cnn_lstm_att_features.shape}")
print(f"cnn_lstm_att_labels shape: {cnn_lstm_att_labels.shape}")


### 加载数据集

from torch.utils.data import TensorDataset, random_split  # 导入PyTorch的数据集和数据划分函数
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    
dataset = TimeSeriesDataset(cnn_lstm_att_features, cnn_lstm_att_labels)

# 定义数据集的大小比例
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

# 计算训练集、验证集和测试集的大小
total_size = len(dataset)  # 总数据量大小
train_size = int(total_size * train_ratio)  # 训练集大小
valid_size = int(total_size * valid_ratio)  # 验证集大小
test_size = total_size - train_size - valid_size  # 测试集大小

train_temp_ratio = train_ratio
test_temp_ratio = 1 - train_ratio
train_temp_size = int(len(dataset) * train_temp_ratio)
test_temp_size = len(dataset) - train_temp_size

# 随机划分数据集
train_dataset, valid_dataset, test_dataset = random_split(
    dataset, [train_size, valid_size, test_size])

# 模型


import sklearn.exceptions
from sklearn.metrics import mean_squared_error,  mean_absolute_error
from torch.optim.lr_scheduler import StepLR
# 定义模型参数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断使用GPU还是CPU进行训练

batch_size = 32
input_len = 128
input_dim =14 # 输入维度
conv_archs = ((1, 16), (1, 32), (1, 64), (1, 128))   # CNN 层卷积池化结构  
hidden_layer_sizes = [16, 32]  # LSTM 层 结构
output_dim = 2  # 输出维度 为2
num_epochs=200
# learn_rate = 0.0000035
# now the best learning_rate = 0.00005
learning_rate = 0.00005
num_heads=2

from mhsa import CNNLSTMAttentionModel
# model = CNNLSTMAttentionModel(batch_size=64, input_dim=3, output_dim=2, conv_archs=[(2, 64)], hidden_layer_sizes=[128], num_heads=4)
model = CNNLSTMAttentionModel(batch_size, input_dim, output_dim, conv_archs, hidden_layer_sizes, num_heads)  
# 定义损失函数和优化函数 
model = model.to(device)
#############定义损失函数############
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)  # 优化器
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


params = list(model.parameters())
k = 0
for idx, i in enumerate(params):
    l = 1
    # print(f"第i层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    # print("该层参数和：" + str(l))
    print(f"第{idx+1}层的结构：{list(i.size())}, 该层参数和：{l}")
    k = k + l
print("总参数数量和：" + str(k))



# 训练：
# from mhsa import calculate_balanced_accuracy, calculate_balanced_f1_score
from mhsa import calculate_accuracy, calculate_f1_score
import time
from sklearn.metrics import classification_report
import sklearn.exceptions
from tqdm import tqdm
train_losses = []
valid_losses = []
train_accuracies = []  # 存储训练准确率
valid_accuracies = []  # 存储验证准确率
train_f1 = []  # 存储训练F1
valid_f1 = []  # 存储验证F1

# 创建DataLoader对象
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
start_time = time.time()  # 记录训练开始时间

for epoch in range(num_epochs):
    model.train()
    total_train_accuracy = 0
    total_train_f1 = 0
    total_train_loss = 0
    start_epoch_time = time.time()  # 记录每轮训练开始时间
    for batch_data, batch_labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        inputs = batch_data.to(device)
        labels = batch_labels.to(device).long()
        outputs = model(inputs)  # (32, 128, 18), outputs: (32, 2)
        labels=labels.reshape(-1)  # (32, )
#         print(outputs.shape)
#         print(labels.shape)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        total_train_accuracy += calculate_accuracy( labels,outputs)
        total_train_f1 += calculate_f1_score(labels,outputs)   
    # scheduler.step()
    end_epoch_time = time.time()  # 记录每轮训练结束时间
    epoch_time_ms = (end_epoch_time - start_epoch_time) * 1000  # 计算每轮训练时间（毫秒）
    print(f'Epoch {epoch + 1}/{num_epochs}, Epoch Time: {epoch_time_ms:.4f} ms')
    
    avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    avg_train_f1 = total_train_f1 / len(train_dataloader)
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_accuracies.append(avg_train_accuracy)
    train_f1.append(avg_train_f1)
    train_losses.append(avg_train_loss)

    # 验证部分
    model.eval()
    total_valid_accuracy = 0
    total_valid_f1 = 0
    total_valid_loss = 0

    with torch.no_grad():
        for (batch_data, batch_labels) in valid_dataloader:
            inputs = batch_data.to(device)
            
            labels = batch_labels.to(device).long()
            labels=labels.reshape(-1)
       
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_valid_loss += loss.item()
            total_valid_accuracy += calculate_accuracy( labels,outputs)
            total_valid_f1 += calculate_f1_score(labels,outputs)
            
    avg_valid_accuracy = total_valid_accuracy / len(valid_dataloader)
    avg_valid_f1 = total_valid_f1 / len(valid_dataloader)
    avg_valid_loss = total_valid_loss / len(valid_dataloader)
    valid_accuracies.append(avg_valid_accuracy)
    valid_f1.append(avg_valid_f1)
    valid_losses.append(avg_valid_loss)

    # 打印训练和验证的统计信息
    print(f'+----------------------Epoch {epoch + 1}/{num_epochs}:--------------------------------+')
    print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Train F1 Score: {avg_train_f1:.4f}')
    print(f'Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}, Valid F1 Score: {avg_valid_f1:.4f}')
    print('+----------------------------------------------------------------------+')
    
    
end_time = time.time()  # 记录训练结束时间
total_training_time_ms = (end_time - start_time) * 1000  # 计算总训练时间（毫秒）
print(f'Total Training Time: {total_training_time_ms:.4f} ms')


# 绘制训练过程中的损失、准确率和F1分数

import matplotlib.pyplot as plt

# 创建一个带有3个子图的图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,6))

# 绘制损失
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(valid_losses, label='Valid Loss')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

# 绘制准确率
axes[1].plot(train_accuracies, label='Train Accuracy')
axes[1].plot(valid_accuracies, label='Valid Accuracy')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

# 绘制F1分数
axes[2].plot(train_f1, label='Train F1 Score')
axes[2].plot(valid_f1, label='Valid F1 Score')
axes[2].set_title('F1 Score')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('F1 Score')
axes[2].legend()
plt.suptitle('Performance Metrics of CNN-LSTM-ATT Model', fontsize=18)

# 调整子图布局
plt.tight_layout()
# plt.show()
plt.savefig('../img/cnn_lstm_att.png')
pass

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
test_preds=[]
test_labels=[]
with torch.no_grad():
    start_time = time.time()  # 记录训练开始时间
    for batch_data, batch_labels in tqdm(test_dataloader):
        inputs = batch_data.to(device)
        labels = batch_labels.to(device).long()
        labels=labels.reshape(-1)
        outputs = model(inputs)
        test_preds.extend(outputs.cpu().detach().numpy())
        test_labels.extend(labels.cpu().detach().numpy())
    end_time = time.time()  # 记录训练结束时间
total_testing_time_ms = (end_time - start_time) * 1000  # 计算总训练时间（毫秒）
test_labels=np.array(test_labels)
test_preds=np.array(test_preds)
test_accuracy = calculate_accuracy(torch.from_numpy(test_labels),torch.from_numpy(test_preds))
test_f1 = calculate_f1_score(torch.from_numpy(test_labels),torch.from_numpy(test_preds))
test_preds_tensor = torch.from_numpy(test_preds) if isinstance(test_preds, np.ndarray) else test_preds
# 使用 torch.max 获取每一行最大值的索引，dim=1 指定沿着列方向
_, predicted_labels = torch.max(test_preds_tensor, dim=1)
predicted_labels=predicted_labels.numpy()
print(f"测试总时间: {total_testing_time_ms:.4f} ms",)
print(f"测试ACC:{test_accuracy}")
print(f"测试F1:{test_f1}")

# 计算混淆矩阵
cm = confusion_matrix(test_labels, predicted_labels)

# 显示混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('../img/confusion_matrix.png')
pass