import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import torch
plt.rcParams['font.sans-serif'] = ['SimHei']#绘图显示中文
plt.rc('font',family='Times New Roman')

data_file = f'../data/full.csv'
df = pd.read_csv(data_file)
df.columns=['islocal','direct','isserver','duration','framesizeration_in','framesizeratio_out','localdelayavg','localdelaynavg',
            'remotedelayavg','remotedelaynavg','dupack','retrans','malf','outoforder','localdelay_label','remotedelay_label','dupack_label',
           'retrans_label','malf_label','outoforder_label']
df_filled = df.fillna(0)

feature_column=df_filled.columns[:14]
# 这里下面对数据的处理能被raw_feature迁移识别吗？
raw_feature=df_filled[feature_column]

pass 
# 对数据进行标准化处理
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
# print(f'df_scale: {df_scale}')  # (2971, 20)

pass

import numpy as np

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

window_size = 16  # 每个窗口包含20个时间点
step_size = 2 # 窗口之间的步长为32个时间点
# localdelay
raw_local = df_scale.iloc[:, :15]

from torch.utils.data import TensorDataset, random_split  # 导入PyTorch的数据集和数据划分函数

from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    
dataset = TimeSeriesDataset(cnn_features, cnn_labels)


# 定义数据集的大小比例
train_ratio = 0.1
valid_ratio = 0.1
test_ratio = 0.8

# 计算训练集、验证集和测试集的大小
total_size = len(dataset)  # 总数据量大小
train_size = int(total_size * train_ratio)  # 训练集大小
valid_size = int(total_size * valid_ratio)  # 验证集大小
test_size = total_size - train_size - valid_size  # 测试集大小

# 随机划分数据集
train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

import torch

def calculate_accuracy(labels,outputs):
    """
    计算准确率。
    outputs: 模型输出，形状为 (batch_size, num_classes)。
    labels: 真实标签，形状为 (batch_size,)。
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def calculate_f1_score(labels,outputs):
    """
    计算F1分数。
    outputs: 模型输出，形状为 (batch_size, num_classes)。
    labels: 真实标签，形状为 (batch_size,)。
    """
    _, predicted = torch.max(outputs, 1)
    
    # 计算TP, FP, FN
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    
    # 计算精确率和召回率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # 计算F1分数
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


