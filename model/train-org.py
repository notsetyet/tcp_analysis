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
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        # 注意: nn.MultiheadAttention的输入形状是 (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # 转置为合适的输入形状
        attn_output, attn_output_weights = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # 再次转置回 (batch_size, seq_len, embed_dim)

        return attn_output, attn_output_weights

##################################################################
##########################CNN-LSTM-ATT##########################
##################################################################
class CNNLSTMAttentionModel(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, conv_archs, hidden_layer_sizes, num_heads=2):
        super().__init__()
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.conv_arch = conv_archs
        self.input_channels = input_dim
        self.cnn_features = self.make_layers()

        self.num_layers = len(hidden_layer_sizes)
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(conv_archs[-1][-1], hidden_layer_sizes[0], batch_first=True))
        for i in range(1, self.num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i-1], hidden_layer_sizes[i], batch_first=True))

        self.attention = MultiHeadAttention(hidden_layer_sizes[-1], num_heads)  # 使用多头注意力

        self.linear = nn.Linear(hidden_layer_sizes[-1], output_dim)

        
      # CNN卷积池化结构
    def make_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_arch:
            for _ in range(num_convs):
                conv = nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1)
                init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
                layers.append(conv)
                # layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def _calculate_l2_regularization(self, weight):
        """
        计算CNN层权重的L2正则化项
        """
        return torch.sum(weight.pow(2))

    def _calculate_l1_regularization(self):
        """
        计算LSTM层权重的L1正则化项
        """
        l1_reg = 0
        for lstm in self.lstm_layers:
            for param in lstm.parameters():
                l1_reg += torch.sum(torch.abs(param))
        return l1_reg


    def forward(self, input_seq):
        # CNN 卷积池化
        # CNN 网络输入[batch,H_in, seq_length]
        # 调换维度[B, L, D] --> [B, D, L]
        input_seq = input_seq.permute(0,2,1)
        cnn_features = self.cnn_features(input_seq) # torch.Size([256, 6, 256])
        # print(cnn_features.size())
#         print(cnn_features.shape)
        # 送入 LSTM 层
        #改变输入形状，lstm 适应网络输入[batch, seq_length, H_in]
        # l2_reg_term = self._calculate_l2_regularization(self.cnn_features.weight)
        lstm_out = cnn_features.permute(0, 2, 1)
        for lstm in self.lstm_layers:
            lstm_out, _= lstm(lstm_out)  ## 进行一次LSTM层的前向传播  

        # l1_reg_term = self._calculate_l1_regularization()    
        attention_out, attention_weights = self.attention(lstm_out)  #  attention
        attention_out = attention_out.mean(dim=1)  # mean
#         print(attention_out.shape)
        predict = self.linear(attention_out)
        # loss_reg = self.cnn_l2_reg * l2_reg_term + self.lstm_l1_reg * l1_reg_term

        # return predict,loss_reg
        return predict



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

def calculate_balanced_accuracy(labels, outputs):
    """
    计算平衡准确率。
    outputs: 模型输出，形状为 (batch_size, num_classes)。
    labels: 真实标签，形状为 (batch_size,)。
    """
    _, predicted = torch.max(outputs, 1)
    unique_classes = unique_labels(labels.cpu().numpy())
    recall_scores_per_class = []
    for class_label in unique_classes:
        # 筛选出属于当前类别的真实标签和预测标签
        current_class_labels = (labels.cpu().numpy() == class_label).astype(int)
        current_class_predicted = (predicted.cpu().numpy() == class_label).astype(int)
        # 计算当前类别的召回率（Recall），在二分类下等同于准确率，多分类下是每个类别的预测准确程度衡量指标之一
        recall = recall_score(current_class_labels, current_class_predicted)
        recall_scores_per_class.append(recall)
    # 计算平衡准确率，即各个类别召回率（准确率）的平均值
    balanced_accuracy = sum(recall_scores_per_class) / len(recall_scores_per_class)
    return balanced_accuracy

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




def load_and_preprocess_data(data_file_path, window_size, step_size, train_ratio, valid_ratio):
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

    feature_column = df_filled.columns[:14]
    raw_feature = df_filled[feature_column]
    localdelay_y = df_filled['localdelay_label']
    remotedelay_y = df_filled['remotedelay_label']
    dupack_y = df_filled['dupack_label']
    retrans_y = df_filled['retrans_label']
    malf_y = df_filled['malf_label']
    outoforder_y = df_filled['outoforder_label']

    scaler = StandardScaler()
    scaled_feature = scaler.fit_transform(raw_feature.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
    df_scale = df_filled.copy()
    df_scale.iloc[:, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]] = scaled_feature

    raw_local = df_scale.iloc[:, :15]
    features, labels = split_time_series(raw_local.values, window_size, step_size)
    cnn_lstm_att_features = torch.tensor(features, dtype=torch.float)
    cnn_lstm_att_labels = torch.tensor(labels, dtype=torch.float)

    dataset = TimeSeriesDataset(cnn_lstm_att_features, cnn_lstm_att_labels)

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

    input_dim = 14  # 根据前面处理逻辑确定的输入维度，可根据实际情况调整提取方式
    output_dim = 2

    return train_dataset, valid_dataset, test_dataset, input_dim, output_dim


def split_time_series(data, window_size, step_size):
    """
    将时间序列数据切分为较短的窗口。

    参数:
    data (np.ndarray): 时间序列数据，形状为 (M, N)，M 是时间点数量，N是特征数量。
    window_size (int): 每个窗口的大小（时间点数量）。
    step_size (int): 窗口之间的步长（时间点数量）。

    返回:
    windows (np.ndarray): 切分后的数据，形状为 (num_windows, window_size, N)。
    labels (np.ndarray): 对应的标签数据，形状为 (num_windows,)。
    """
    M, N = data.shape
    num_windows = (M - window_size) // step_size + 1
    windows = np.empty((num_windows, window_size, N - 1))
    labels = []

    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx, :-1]

        labels.append(data[end_idx - 1, -1])

    return windows, np.array(labels)


class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_model(train_dataset, valid_dataset, input_dim, output_dim, num_epochs, batch_size, learning_rate):
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
    conv_archs = ((1, 16), (1, 32), (1, 64), (1, 128))
    hidden_layer_sizes = [16, 32]
    num_heads = 2
    model = CNNLSTMAttentionModel(batch_size, input_dim, output_dim, conv_archs, hidden_layer_sizes, num_heads)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    train_f1 = []
    valid_f1 = []

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        total_train_accuracy = 0
        total_train_f1 = 0
        total_train_loss = 0

        for batch_data, batch_labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            inputs = batch_data.to(device)
            labels = batch_labels.to(device).long()
            outputs = model(inputs)
            labels = labels.reshape(-1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_accuracy += calculate_accuracy(labels, outputs)
            total_train_f1 += calculate_f1_score(labels, outputs)

        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        avg_train_f1 = total_train_f1 / len(train_dataloader)
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracies.append(avg_train_accuracy)
        train_f1.append(avg_train_f1)
        train_losses.append(avg_train_loss)

        model.eval()
        total_valid_accuracy = 0
        total_valid_f1 = 0
        total_valid_loss = 0

        with torch.no_grad():
            for (batch_data, batch_labels) in valid_dataloader:
                inputs = batch_data.to(device)
                labels = batch_labels.to(device).long()
                labels = labels.reshape(-1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()
                total_valid_accuracy += calculate_accuracy(labels, outputs)
                total_valid_f1 += calculate_f1_score(labels, outputs)

        avg_valid_accuracy = total_valid_accuracy / len(valid_dataloader)
        avg_valid_f1 = total_valid_f1 / len(valid_dataloader)
        avg_valid_loss = total_valid_loss / len(valid_dataloader)
        valid_accuracies.append(avg_valid_accuracy)
        valid_f1.append(avg_valid_f1)
        valid_losses.append(avg_valid_loss)

        print(f'+----------------------Epoch {epoch + 1}/{num_epochs}:--------------------------------+')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Train F1 Score: {avg_train_f1:.4f}')
        print(f'Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}, Valid F1 Score: {avg_valid_f1:.4f}')
        print('+----------------------------------------------------------------------+')

    # model_save_path='./'
    # with open(model_save_path, 'wb') as f:
    #     pickle.dump(model, f)
    return train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1


def plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1, save_path):
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
    fig.suptitle('Performance Metrics of CNN-LSTM-ATT Model', fontsize=18)

    fig.tight_layout()
    fig.savefig(save_path)


def test_model(model, test_dataset, batch_size, device, save_path):
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

def  get_CNNLSTMAttentionModel(batch_size, input_dim, output_dim):
    conv_archs = ((1, 16), (1, 32), (1, 64), (1, 128))
    hidden_layer_sizes = [16, 32]
    num_heads = 2
    return CNNLSTMAttentionModel(batch_size, input_dim, output_dim, conv_archs, hidden_layer_sizes, num_heads)

if __name__ == "__main__":
    data_file_path = '../data/output.csv'
    window_size = 16
    step_size = 2
    train_ratio = 0.6
    valid_ratio = 0.2
    num_epochs = 200
    batch_size = 32
    learning_rate = 0.00005
    model_save_path = '../'  # 假设保存模型的路径，可根据实际调整
    img_save_path = '../img/'

    train_dataset, valid_dataset, test_dataset, input_dim, output_dim = load_and_preprocess_data(data_file_path,
                                                                                                window_size,
                                                                                                step_size,
                                                                                                train_ratio,
                                                                                                valid_ratio)
    train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1 = train_model(train_dataset,
                                                                                                    valid_dataset,
                                                                                                    input_dim,
                                                                                                    output_dim,
                                                                                                    num_epochs,
                                                                                                    batch_size,
                                                                                                    learning_rate)
    plot_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, train_f1, valid_f1,
                 img_save_path + 'cnn_lstm_att.png')
    

    model = get_CNNLSTMAttentionModel(batch_size, input_dim, output_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model(model, test_dataset, batch_size, device, img_save_path + 'cnn_lstm_att_confusion_matrix.png')
    # 这里可以添加