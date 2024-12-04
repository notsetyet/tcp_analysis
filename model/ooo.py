import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
# from mhsa import CNNLSTMAttentionModel
# from mhsa import calculate_accuracy, calculate_f1_score
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import pickle





import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
import matplotlib

matplotlib.use('Agg')
##################################################################
##########################多头注意力机制##########################
##################################################################
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        # 调整hx和cx的维度
        x = x.view(len(x), 1, -1) 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        # print(out.shape)
        out = self.fc(out[:, -1, :])
        # print(out.shape)
        return out


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




def load_and_preprocess_data(data_file_path):
    """
    加载数据、进行预处理、划分时间序列窗口并划分数据集。

    参数:
    data_file_path (str): 数据文件的路径。
    window_size (int): 每个时间序列窗口的大小。
    step_size (int): 窗口之间的步长。
    train_ratio (float): 训练集所占比例。
    valid_ratio (float): 验证集所占比例。

    返回:
    train_dataset (Dataset): 训练数据集。
    valid_dataset (Dataset): 验证数据集。
    test_dataset (Dataset): 测试数据集。
    input_dim (int): 输入维度。
    output_dim (int): 输出维度。
    """
    df = pd.read_csv(data_file_path)
    df.columns = ['islocal', 'direct', 'isserver', 'duration', 'framesizeration_in', 'framesizeratio_out',
                  'localdelayavg', 'localdelaynavg',
                  'remotedelayavg', 'remotedelaynavg', 'dupack', 'retrans', 'malf', 'outoforder', 'localdelay_label',
                  'remotedelay_label', 'dupack_label',
                  'retrans_label', 'malf_label', 'outoforder_label']
    df_filled = df.fillna(0)

    feature_column = df.columns[:14]
    raw_feature = df[feature_column]
    scaler = StandardScaler()
    scaled_feature = scaler.fit_transform(raw_feature.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    df_scale = df_filled
    df_scale.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = scaled_feature


    return df_scale

def load_ooo_data(df, train_ratio, valid_ratio):
    # df = data_file
    # feature_column = df.columns[:14]
    # raw_feature = df[feature_column]
    # scaler = StandardScaler()
    # scaled_feature = scaler.fit_transform(raw_feature.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    # df_scale = df.copy()
    # df_scale.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = scaled_feature

    # raw_local = df_scale.iloc[:, :15]
    # features, labels = split_time_series(raw_local.values, window_size, step_size)
    # cnn_lstm_att_features = torch.tensor(features, dtype=torch.float)
    # cnn_lstm_att_labels = torch.tensor(labels, dtype=torch.float).reshape(-1, 1)

    # dataset = TimeSeriesDataset(cnn_lstm_att_features, cnn_lstm_att_labels)

    raw_feature=df.iloc[:, :14]
    local_label=df.iloc[:, 19:20]
    X_scaled_tensor = torch.tensor(raw_feature.values, dtype=torch.float32)  # 将标准化后的数据转换为PyTorch张量
    y_tensor = torch.tensor(local_label.values, dtype=torch.int64).reshape(-1, 1)  # 将目标列转换为PyTorch张量，并调整形状以匹配预期格式

# 创建Dataset对象
    dataset = CustomDataset(X_scaled_tensor, y_tensor)

    # # 计算训练集、验证集和测试集的大小
    # total_size = len(dataset)  # 总数据量大小
    # train_size = int(total_size * train_ratio)  # 训练集大小
    # valid_size = int(total_size * valid_ratio)  # 验证集大小
    # test_size = total_size - train_size - valid_size  # 测试集大小

    # train_temp_ratio = train_ratio
    # test_temp_ratio = 1 - train_ratio
    # train_temp_size = int(len(dataset) * train_temp_ratio)
    # test_temp_size = len(dataset) - train_temp_size

    # # 随机划分数据集
    # train_dataset, valid_dataset, test_dataset = random_split(
    #     dataset, [train_size, valid_size, test_size])

    # 计算训练集、验证集和测试集的大小
    total_size = len(y_tensor)  # 总数据量大小
    train_size = int(total_size * train_ratio)  # 计算训练集大小
    valid_size = int(total_size * valid_ratio)  # 计算验证集大小
    test_size = total_size - train_size - valid_size  # 计算测试集大小，确保三者之和等于总数据量

    # 随机划分数据集
    train_dataset, temp_dataset = random_split(dataset, [train_size, total_size - train_size])  # 先将数据分为训练集和临时数据集（后者包含验证集和测试集）
    valid_dataset, test_dataset = random_split(temp_dataset, [valid_size, test_size])  # 再将临时数据集分为验证集和测试集


    input_dim = 14  # 根据前面处理逻辑确定的输入维度，可根据实际情况调整提取方式
    output_dim = 2

    return train_dataset, valid_dataset, test_dataset, input_dim, output_dim



class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_outoforder(train_dataset, valid_dataset, input_dim, output_dim, num_epochs, batch_size, learning_rate):
    """
    训练模型，包括定义模型、损失函数、优化器，进行训练循环并返回训练过程中的统计信息。

    参数:
    train_dataset (Dataset): 训练数据集。
    valid_dataset (Dataset): 验证数据集。
    input_dim (int): 输入维度。
    output_dim (int): 输出维度。
    num_epochs (int): 训练的轮数。
    batch_size (int): 每批次数据的大小。
    learning_rate (float): 学习率。

    返回:
    train_losses (list): 每轮训练的损失值列表。
    valid_losses (list): 每轮验证的损失值列表。
    train_accuracies (list): 每轮训练的准确率列表。
    valid_accuracies (list): 每轮验证的准确率列表。
    train_f1 (list): 每轮训练的F1分数列表。
    valid_f1 (list): 每轮验证的F1分数列表。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_train_losses = []
    lstm_valid_losses = []
    lstm_train_accuracies = []  # 存储训练准确率
    lstm_valid_accuracies = []  # 存储验证准确率
    lstm_train_f1 = []  # 存储训练F1
    lstm_valid_f1 = []  # 存储验证F1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断使用GPU还是CPU进行训练
     
    model = BiLSTM(input_dim, 64, 1,output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 创建DataLoader对象
    lstm_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    lstm_valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # lstm_test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(num_epochs):
        model.train()
        total_train_accuracy = 0
        total_train_f1 = 0
        total_train_loss = 0

        for (batch_data, batch_labels)in lstm_train_dataloader:
            inputs = batch_data.to(device)
            labels = batch_labels.to(device).long()
            
            outputs = model(inputs)
            labels=labels.reshape(-1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_accuracy += calculate_accuracy( labels,outputs)
            total_train_f1 += calculate_f1_score(labels,outputs)
       
        avg_train_accuracy = total_train_accuracy / len(lstm_train_dataloader)
        avg_train_f1 = total_train_f1 / len(lstm_train_dataloader)
        avg_train_loss = total_train_loss / len(lstm_train_dataloader)
        lstm_train_accuracies.append(avg_train_accuracy)
        lstm_train_f1.append(avg_train_f1)
        lstm_train_losses.append(avg_train_loss)

    # 验证部分
        model.eval()
        total_valid_accuracy = 0
        total_valid_f1 = 0
        total_valid_loss = 0

        with torch.no_grad():
            for (batch_data, batch_labels) in lstm_valid_dataloader:
                inputs = batch_data.to(device)
                
                labels = batch_labels.to(device).long()
                labels=labels.reshape(-1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()
                total_valid_accuracy += calculate_accuracy( labels,outputs)
                total_valid_f1 += calculate_f1_score(labels,outputs)
            
        avg_valid_accuracy = total_valid_accuracy / len(lstm_valid_dataloader)
        avg_valid_f1 = total_valid_f1 / len(lstm_valid_dataloader)
        avg_valid_loss = total_valid_loss / len(lstm_valid_dataloader)
        lstm_valid_accuracies.append(avg_valid_accuracy)
        lstm_valid_f1.append(avg_valid_f1)
        lstm_valid_losses.append(avg_valid_loss)

        # 打印训练和验证的统计信息
        print(f'+----------------------Epoch {epoch + 1}/{num_epochs}:--------------------------------+')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Train F1 Score: {avg_train_f1:.4f}')
        print(f'Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}, Valid F1 Score: {avg_valid_f1:.4f}')
        print('+----------------------------------------------------------------------+')

    # model_save_path='./'
    # with open(model_save_path, 'wb') as f:
    #     pickle.dump(model, f)
    return lstm_train_losses, lstm_valid_losses, lstm_train_accuracies, lstm_valid_accuracies, lstm_train_f1, lstm_valid_f1

def  get_LSTMModel(input_dim, output_dim):
    # batch_size, input_size, hidden_size, num_layers, num_classes
    return BiLSTM(input_dim, 64, 1, output_dim)

def plot_ooo(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1, save_path):
    """
    绘制训练过程中的损失、准确率和F1分数图像并保存。

    参数:
    train_losses (list): 每轮训练的损失值列表。
    valid_losses (list): 每轮验证的损失值列表。
    train_accuracies (list): 每轮训练的准确率列表。
    valid_accuracies (list): 每轮验证的准确率列表。
    train_f1 (list): 每轮训练的F1分数列表。
    valid_f1 (list): 每轮验证的F1分数列表。
    save_path (str): 图像保存的路径。
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

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
    plt.suptitle('OutOfOrder Performance of CNN-LSTM-ATT Model', fontsize=18)

    plt.tight_layout()
    plt.savefig(save_path)


def test_ooo(model, test_dataset, batch_size, device, save_path):
    """
    使用测试数据集对训练好的模型进行测试，计算相关指标并绘制混淆矩阵保存。

    参数:
    model (nn.Module): 训练好的模型。
    test_dataset (Dataset): 测试数据集。
    batch_size (int): 每批次数据的大小。
    device (torch.device): 运行设备（GPU或CPU）。
    save_path (str): 混淆矩阵图像保存的路径。

    返回:
    test_accuracy (float): 测试集准确率。
    test_f1 (float): 测试集F1分数。
    """
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_preds = []
    test_labels = []
    with torch.no_grad():
        start_time = time.time()
        for batch_data, batch_labels in tqdm(test_dataloader):
            inputs = batch_data.to(device)
            labels = batch_labels.to(device).long()
            labels = labels.reshape(-1)
            outputs = model(inputs)
            test_preds.extend(outputs.cpu().detach().numpy())
            test_labels.extend(labels.cpu().detach().numpy())
        end_time = time.time()
    total_testing_time_ms = (end_time - start_time) * 1000
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_accuracy = calculate_accuracy(torch.from_numpy(test_labels), torch.from_numpy(test_preds))
    test_f1 = calculate_f1_score(torch.from_numpy(test_labels), torch.from_numpy(test_preds))
    test_preds_tensor = torch.from_numpy(test_preds) if isinstance(test_preds, np.ndarray) else test_preds
    _, predicted_labels = torch.max(test_preds_tensor, dim=1)
    predicted_labels = predicted_labels.numpy()

    print(f"测试总时间: {total_testing_time_ms:.4f} ms")
    print(f"测试ACC:{test_accuracy}")
    print(f"测试F1:{test_f1}")

    cm = confusion_matrix(test_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(save_path)

    return test_accuracy, test_f1


if __name__ == "__main__":
    data_file_path = '../data/output.csv'
    window_size = 16
    step_size = 2
    train_ratio = 0.8
    valid_ratio = 0.1
    num_epochs = 30
    batch_size = 1024
    learning_rate = 0.001
    model_save_path = '../'  # 假设保存模型的路径，可根据实际调整
    img_save_path = '../img/'

    df = load_and_preprocess_data(data_file_path)
    train_dataset, valid_dataset, test_dataset, input_dim, output_dim = load_ooo_data(df, train_ratio, valid_ratio)

    train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1 = train_outoforder(train_dataset,
                                                                                                    valid_dataset,
                                                                                                    input_dim,
                                                                                                    output_dim,
                                                                                                    num_epochs,
                                                                                                     batch_size,
                                                                                                    learning_rate)

    # train_file = img_save_path + 'ooo.png'
    # plot_ooo(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1,
                #  train_file)
    # model = CNNLSTMAttentionModel(batch_size, input_dim, output_dim, conv_archs, hidden_layer_sizes, num_heads)
    # 这里可以添加