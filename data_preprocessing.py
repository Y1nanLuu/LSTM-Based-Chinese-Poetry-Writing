import re
import numpy as np
import pickle
import os
from torch.utils.data import Dataset, DataLoader

def preprocess_text(text):
    """预处理文本，删除题目、括号内容和书名号内容"""
    # 删除每行开头到冒号的内容（题目）
    text = re.sub(r'^[^:]+:', '', text, flags=re.MULTILINE)
    # 删除括号内的内容，包括各种括号
    text = re.sub(r'[\(\)\[\]\{\}（）【】『』]', '', text)
    # 删除书名号及其中间的内容
    text = re.sub(r'《.*?》', '', text)
    # 去除空行
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def load_and_preprocess_data(file_path, output_dir='preprocessed_data'):
    """加载数据并进行预处理"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 检查是否已有预处理好的数据
    preprocessed_file = os.path.join(output_dir, 'preprocessed_text.txt')
    if os.path.exists(preprocessed_file):
        with open(preprocessed_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    # 读取原始数据
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 预处理
    processed_text = preprocess_text(text)
    
    # 保存预处理结果
    with open(preprocessed_file, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    
    return processed_text

def create_vocabulary(text, output_dir='preprocessed_data'):
    """创建词汇表"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    vocab_file = os.path.join(output_dir, 'vocab.pkl')
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as f:
            return pickle.load(f)
    
    # 收集所有字符
    chars = sorted(list(set(text)))
    char_to_idx = {char: i for i, char in enumerate(chars)}
    idx_to_char = {i: char for i, char in enumerate(chars)}
    vocab_size = len(chars)
    
    vocab = {
        'chars': chars,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }
    
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f)
    
    return vocab

class PoetryDataset(Dataset):
    """诗歌数据集"""
    def __init__(self, text, vocab, seq_length=50):
        self.text = text
        self.vocab = vocab
        self.seq_length = seq_length
        self.char_to_idx = vocab['char_to_idx']
        self.data = self._prepare_sequences()
    
    def _prepare_sequences(self):
        """准备训练序列"""
        sequences = []
        targets = []
        
        for i in range(0, len(self.text) - self.seq_length):
            seq_in = self.text[i:i + self.seq_length]
            seq_out = self.text[i + self.seq_length]
            sequences.append([self.char_to_idx[char] for char in seq_in])
            targets.append(self.char_to_idx[seq_out])
        
        return list(zip(sequences, targets))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence, target = self.data[idx]
        return (np.array(sequence, dtype=np.int64), 
                np.array(target, dtype=np.int64))

def get_data_loader(text, vocab, seq_length=50, batch_size=32, shuffle=True):
    """获取数据加载器"""
    dataset = PoetryDataset(text, vocab, seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)