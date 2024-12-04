

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.multiclass import unique_labels

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

def calculate_balanced_f1_score(labels, outputs):
    """
    计算F1分数（采用宏观平均方式，适用于多分类场景，考虑类别不平衡情况）。
    outputs: 模型输出，形状为 (batch_size, num_classes)。
    labels: 真实标签，形状为 (batch_size,)。
    """
    _, predicted = torch.max(outputs, 1)
    unique_classes = torch.unique(labels).tolist()
    f1_scores_per_class = []
    for class_label in unique_classes:
        # 筛选出属于当前类别的真实标签和预测标签
        current_class_labels = (labels == class_label).long()
        current_class_predicted = (predicted == class_label).long()

        # 计算当前类别的精确率
        precision = precision_score(current_class_labels.cpu().numpy(),
                                    current_class_predicted.cpu().numpy(),
                                    zero_division=0)

        # 计算当前类别的召回率
        recall = recall_score(current_class_labels.cpu().numpy(),
                              current_class_predicted.cpu().numpy(),
                              zero_division=0)

        # 计算当前类别的F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores_per_class.append(f1)

    # 计算宏观平均F1分数（对所有类别的F1分数求平均）
    macro_f1_score = sum(f1_scores_per_class) / len(f1_scores_per_class)
    return macro_f1_score