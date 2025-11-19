import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class PositionalEncoding(nn.Module):
    """位置编码：为序列添加位置信息，Transformer必需组件"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度用正弦，奇数维度用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 形状: (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # 不参与训练的参数
    
    def forward(self, x):
        # x形状: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]  # 加入位置编码
        return self.dropout(x)


class PoetryRNN(nn.Module):
    """基于LSTM的诗歌生成模型"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(PoetryRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x形状: (batch_size, seq_length)
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        # 嵌入层
        x_embed = self.embedding(x)  # 形状: (batch_size, seq_length, embed_dim)
        
        # LSTM层
        lstm_out, hidden = self.lstm(x_embed, hidden)  # lstm_out形状: (batch_size, seq_length, hidden_dim)
        
        # 取最后一个时间步的输出用于预测
        # 或者使用所有时间步的输出进行训练
        output = self.fc(lstm_out)  # 形状: (batch_size, seq_length, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

class PoetryBiRNN(nn.Module):
    """基于BiLSTM的诗歌生成模型"""
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super(PoetryBiRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)  # 新增：嵌入层归一化

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # 双向LSTM（bidirectional=True）
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True  # 核心：双向建模
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.bifc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)  # 新增：输出前dropout正则化
        
    def forward(self, x, hidden=None):
        # x形状: (batch_size, seq_length)
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        
        # 嵌入层
        x_embed = self.embedding(x)  # 形状: (batch_size, seq_length, embed_dim)
        x_embed = self.embed_norm(x_embed)  # 归一化
        x_embed = self.dropout(x_embed)  # 正则化

        # LSTM层
        lstm_out, hidden = self.bilstm(x_embed, hidden)  # lstm_out形状: (batch_size, seq_length, hidden_dim)
        
        # 取最后一个时间步的输出用于预测
        # 或者使用所有时间步的输出进行训练
        output = self.bifc(lstm_out)  # 形状: (batch_size, seq_length, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

def save_model(model, path):
    """保存模型"""
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """加载模型参数"""
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model