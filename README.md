# LSTM-Based-Chinese-Poetry-Writing

**written by YinanLuu
2025.11.19**

## About training and generating

### 训练

下面的命令可以在12G的显卡上运行，但是可能需要检查一下cuda编号。

```bash
python train.py --data_path data/raw_poems.txt --epochs 10 --batch_size 512 --lr 0.001 --model_dir Bimodels 
```

定位到下面代码处可以修改训练单向LSTM或双向LSTM。如果训练不同的模型注意把checkpoint存储到不同目录下，单向models，双向Bimodels

```python
model = PoetryBiRNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
```

### 生成

同样要针对不同模型修改代码

```bash
python main.py --type normal --length 4 --model_path Bimodels/best_model.pth # 普通诗歌
python main.py --type acrostic --head_chars "我自风来" --lines_per_head 1 --temperature 0.6 # 藏头诗
python main.py --type long --length 30 --gpu_id 6 --temperature 0.8 # 超长诗
```

## About experiment report

## 一、实验目标

1. 训练一个循环神经网络，实现AI写诗

2. 理解循环神经网络的基本结构、代码实现及训练过程

3. 比较RNN、LSTM的不同之处

4. 生成藏头诗、超长诗（超过20句）、自定义主题或设置，并作结果分析

   


## 二、实验环境

​	实验环境配置如下：代码开发采用 Visual Studio Code，深度学习框架选用 PyTorch，并在 Conda 虚拟环境中运行。模型训练过程使用笔记本配备的 NVIDIA RTX 4050 显卡进行加速。具体软件版本如下所示：

- Python 版本：3.11.5
- PyTorch 版本：2.8.0+cu126
- Torchvision 版本：0.23.0+cu126



## 三、实验原理

### （一）循环神经网络

​	**循环神经网络（Recurrent Neural Network, RNN）**是一类能够处理序列数据的神经网络结构。与传统前馈网络不同，RNN 在时间维度上具有“记忆”能力，它通过隐藏状态（hidden state）将过去的信息传递到当前，从而能够在输入序列中捕捉时间依赖关系。

传统 RNN 的核心是：
$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$


​	它通过简单的循环结构将前一时刻的状态$h_{t-1}$ 与当前输入 $x_t$ 结合。但由于梯度消失导致无法学习较长序列的信息。

​	但是，**传统 RNN 容易出现梯度消失和梯度爆炸**的问题，使得网络在处理长期依赖时效果不佳。为了解决这一问题，出现了改进版本，如 **LSTM** 和 **GRU**。

​	**长短期记忆网络（LSTM）** 是为解决传统 RNN 无法有效捕捉长期依赖的问题而提出的一种结构改进。它通过引入遗忘门、输入门和输出门等门控机制，对信息的流入、保留与输出进行精确调控。这样，模型可以在较长的时间跨度上保留有用特征，避免梯度消失导致的记忆衰减。尽管 LSTM 具有更强的表达能力，但其内部结构相对复杂，参数量较大，在训练速度和资源消耗方面会略逊于更轻量的循环模型。

​	**门控循环单元（GRU）** 则是在 LSTM 的基础上进一步简化门控设计的一种结构。它将遗忘门和输入门合并为更新门，并利用重置门控制历史信息在生成当前状态时的作用，使得模型在保持长期依赖能力的同时降低了结构复杂度。由于参数更少、计算更高效，GRU 在许多任务中的表现接近甚至不逊于 LSTM，特别适合需要实时响应或资源受限的应用场景。



## 四、实验过程

### （一）实验数据预处理

#### 1、清理数据

​	实验原始数据是如下的txt文档，结构较混乱。大部分诗歌以行为单位，冒号前是标题。但也有部分诗歌有括号内注释，诗人名等信息。并且诗的体裁，长度等都有区别。
<img width="1333" height="219" alt="屏幕截图 2025-11-17 130700" src="https://github.com/user-attachments/assets/692cd403-d2bd-4d38-b911-f5e4a7faba5e" />

```python
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
```

通过清理数据集，将标题以及括号内、书名号内的内容全部清除，得到较为干净的数据集。


#### 2、制作词表	

​	由于汉字不能直接输入到模型中，需要制作词表，将汉字与index一一对应。先从清理好的文本中提取所有不同的字符集合，然后转换为列表，再按照unicode编码顺序排序得到有序词表。

​	在训练阶段，需要将字符映射到index，生成阶段，需要将index映射到字符，因此vocab词表需要有char_to_idx和idx_to_char两个映射。

```python
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
```

#### 3、封装Dataset类创建Dataloader

​	核心逻辑是从原始文本中截取固定长度的输入序列，并对应一个目标字符。这里的seq_lenth默认设置为50，也就是类似与用前50个字符预测第51个字符。

​	`_prepare_sequences`方法遍历文本，每次截取长度为seq_lenth的序列作为输入，将第seq_lenth+1个字符作为输出。利用词表将文本序列转化为索引序列，最后返回列表（每个元素是(sequence, target)元组）。

​	`__getitem__`方法按照index获取列表中的(sequence, target)元组。这是一个训练样本。

​	值得注意的是，清理文本时没有去掉标点，所以预测标点何时生成也是训练目标之一。

```python
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
```

### （二）模型设计、训练与生成

#### 1、模型

##### （1）基础模型

​	基础模型采用单向LSTM。由一层嵌入层（嵌入维度embed_dim=128），两层LSTM（num_layers=2，隐藏层维度hidden_dim=256），一层全连接（用于将隐藏层映射到输出），dropout=0.2。

​	初始化隐藏状态使用零初始化。

​	LSTM的隐藏状态是用来保存序列的上下文信息的。h0记录当前时刻的输出信息，用于传递短期即系。c0是细胞状态，用于传递长期记忆，通过门控机制控制信息的增减。这两个状态的初始值会直接影响模型对序列的处理，尤其是在处理第一个输入时（此时尚无历史信息）。

```python
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

```

##### （2）双向LSTM

​	双向LSTM依旧相比单向LSTM，新增了嵌入层归一化、dropout归一化，此外针对双向进行了维度的适应调整，大体没太变化。

```python
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

```


#### 2、训练

​	没啥特别的。但是在不同的训练过程中使用了不同的学习率调度器，例如自适应调度器和指数衰减调度器。

#### 3、生成诗歌

​	采样函数sample，用于将模型前向生成的概率分布转化为具体的字符索引。model输出是一个vocab_size大小的向量，里面每个维度都是这个索引的字的概率。**用temperature调节随机性**。

- `temperature > 1`：概率分布更平缓（降低高概率字符的优势），生成更随机、更多样。
- `temperature < 1`：概率分布更陡峭（放大高概率字符的优势），生成更保守、更稳定。
- `temperature = 1`：直接使用模型输出的概率分布。

```python
def sample(preds, temperature=1.0):
    """根据预测结果和温度生成下一个字符的索引"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

​	生成函数generate_text，以种子文本为起点，每次预测并添加一个字符，通过滑动窗口更新输入，直到满足自定义的`stop_condition`（停止条件）。

​	**种子文本seed_text**是指用来生成最初的字的那些文本。可以在词表中随机采样。对于藏头诗，第一句的种子文本是头的第一个字，后续的种子文本是上一句生成的内容。

​	**停止条件stop_condition**一般是指达到了生成的句数要求。在后续生成代码中，将句号视为一行结束的标志。

```python
def generate_text(model, vocab, seed_text, stop_condition, temperature=0.7, 
                  seq_length=50, device='cpu'):
    """生成文本（适配动态停止条件）"""
    char_to_idx = vocab['char_to_idx']
    idx_to_char = vocab['idx_to_char']
    vocab_size = vocab['vocab_size']
    
    # 确保种子文本长度符合要求
    if len(seed_text) > seq_length:
        seed_text = seed_text[-seq_length:]
    elif len(seed_text) < seq_length:
        seed_text = ' ' * (seq_length - len(seed_text)) + seed_text
    
    seed_indices = [char_to_idx.get(char, 0) for char in seed_text]
    generated = seed_text.lstrip()  # 去除开头填充的空格
    
    # 初始化隐藏状态
    model.eval()
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        # 处理种子文本获取初始隐藏状态
        seed_tensor = torch.tensor([seed_indices], dtype=torch.long).to(device)
        _, hidden = model(seed_tensor, hidden)
        
        # 生成逻辑：直到满足停止条件
        while not stop_condition(generated):
            # 取最后一个字符作为输入
            input_tensor = torch.tensor([[seed_indices[-1]]], dtype=torch.long).to(device)
            output, hidden = model(input_tensor, hidden)
            output = torch.softmax(output[0, 0], dim=0).cpu().numpy()
            
            # 采样下一个字符
            next_idx = sample(output, temperature)
            next_char = idx_to_char.get(next_idx, '')
            generated += next_char
            seed_indices = seed_indices[1:] + [next_idx]
            
            # 防止无限循环（最大生成字符数上限）
            if len(generated) > 1000:
                break
    
    return generated

```

​	普通诗歌生成函数generate_poem，调用generate_text，生成一个句号换一次行，用line_count控制生成的行数。停止条件是句数到达限制，如果没有种子文本，随机采样词表中的汉字。

```python
def generate_poem(model, vocab, seed_text=None, line_count=20, temperature=0.7, 
                  seq_length=50, device='cpu'):
    """生成普通诗（按句号换行，达到指定行数停止）"""
    if not seed_text:
        seed_text = np.random.choice(list(vocab['chars']))
    
    # 停止条件：生成的文本中句号数量 >= 目标行数
    def stop_condition(text):
        period_count = text.count('。')
        return period_count >= line_count
    
    # 生成文本
    poem_text = generate_text(model, vocab, seed_text, stop_condition, 
                             temperature, seq_length, device)
    
    # 按句号分割，格式化分行（去除空行）
    poem_lines = [line.strip() + '。' for line in poem_text.split('。') if line.strip()]
    # 确保行数匹配（可能多生成了一行，截取前N行）
    poem_lines = poem_lines[:line_count]
    
    return '\n'.join(poem_lines)
```

​	生成藏头诗generate_acrostic_poem，输入头字符，用lines_per_head控制一个头字符可以生成多少行诗。

​	传统的藏头诗是，每个逗号都用一个藏头。这里我希望生成的诗歌更长一些，更有意义一些，因此没有遵循传统的设置。

```python
def generate_acrostic_poem(model, vocab, head_chars, temperature=0.7, 
                          seq_length=50, lines_per_head=2, device='cpu'):
    """生成藏头诗（按句号换行，每个藏头字对应指定行数）"""
    poem_lines = []
    total_target_lines = len(head_chars) * lines_per_head
    
    for idx, head_char in enumerate(head_chars):
        # 已生成的行数
        generated_lines = len(poem_lines)
        # 剩余需要为当前藏头字生成的行数
        need_lines = min(lines_per_head, total_target_lines - generated_lines)
        
        for i in range(need_lines):
            if i == 0:
                # 第一句必须以藏头字开头
                seed_text = head_char
                # 停止条件：生成包含至少一个句号的句子
                def stop_condition(text):
                    return '。' in text and len(text) >= 5  # 确保句子长度合理
                
                line = generate_text(model, vocab, seed_text, stop_condition, 
                                    temperature, seq_length, device)
                # 确保以藏头字开头，且只保留第一个句号前的内容
                if line.startswith(head_char):
                    line = line.split('。')[0].strip() + '。'
                else:
                    line = head_char + line[1:].split('。')[0].strip() + '。'
            else:
                # 后续句子基于前一句生成，按句号分割
                seed_text = poem_lines[-1] if poem_lines else head_char
                def stop_condition(text):
                    return '。' in text and len(text) >= 5
                
                line = generate_text(model, vocab, seed_text, stop_condition, 
                                    temperature, seq_length, device)
                line = line.split('。')[0].strip() + '。'
            
            if line.strip():
                poem_lines.append(line)
    
    # 确保总行数符合要求
    poem_lines = poem_lines[:total_target_lines]
    return '\n'.join(poem_lines)

```

​	生成大于20行的长诗，只有强制行数>=21，其他和生成普通诗歌一样。

```python
def generate_long_poem(model, vocab, seed_text=None, line_count=50, temperature=0.7, 
                      seq_length=50, device='cpu'):
    """生成超长诗（按句号换行，行数>=21）"""
    # 确保超长诗行数>=21
    line_count = max(line_count, 21)
    return generate_poem(model, vocab, seed_text, line_count, temperature, 
                        seq_length, device)
```



## 五、实验结果

#### 1、基础模型+学习率自适应调度器

​	对基础模型的训练设置：batch_size=512，epoch=5，lr=0.01。由于设置的学习率过大，导致出现了收敛失败（在后续训练到30个epoch的时候loss再也没下降到6以下），因此取第3个epoch的模型进行诗歌生成实验。

<img width="1301" height="406" alt="c5576efae4045c6c9579b0f900b797bf" src="https://github.com/user-attachments/assets/50300664-35af-4043-9e2c-4271b29010bf" />


​	同时自适应调度器的设置如下。连续三轮损失不下降则调整。但是对于小epoch的训练来说这个调度器形同虚设，因此在后续实验考虑把这个选项优化掉。

```python
    # 初始化学习率调度器（去掉verbose参数，适配低版本PyTorch）
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',          # 针对损失（越小越好）
        factor=0.5,          # 学习率调整倍数（乘以0.5）
        patience=3,          # 3轮损失不下降则调整
        min_lr=1e-6          # 最小学习率（避免过低）
    )

```

​	生成诗歌如下。

##### （1）普通诗歌

​	参数解析：normal普通诗歌，lenth行数，temperature这里没设置，但是默认是0.7，也就是偏保守。可见生成的诗歌比较胡言乱语，句式也不太统一。

<img width="1012" height="418" alt="屏幕截图 2025-11-18 224544" src="https://github.com/user-attachments/assets/b61d01ea-57c2-4f0b-8c8c-57ea263629cf" />


<img width="1016" height="426" alt="屏幕截图 2025-11-18 224611" src="https://github.com/user-attachments/assets/f2c458aa-51aa-4bf8-8f88-96e4e01ff92c" />


​	使用诗歌校验工具[律诗校验](https://sou-yun.cn/analyzepoem.aspx)进行**平仄**和**用韵**，发现机器还是学到了平仄和韵脚相关的知识。其中标**红**处表示平仄有误。标**紫红色**处表示用韵错误。

<img width="447" height="187" alt="屏幕截图 2025-11-19 151015" src="https://github.com/user-attachments/assets/32e0382d-59c3-4d80-b0bd-f7bf3dfadc02" />


​	认为平仄和韵脚这种比较规整的规则，是比较容易被机器学习到的。但是语义特征相对困难，少epoch和简单结构可能对机器来说比较困难。句式方面，可能由于数据集中的诗歌比较多样化，包括五言、七言、诗经体与古体诗，所以生成的内容也比较随机。但是大体上，一句话的上下两句是比较工整的。

##### （2）藏头诗

​	type标识是藏头诗，head_char和lines_per_head共同决定生成的行数，temperature控制生成的比较保守。

​	由于藏头诗生成时强制每句第一个字，诗歌的上下文语义会被影响，但是并非完全不受之前的句子影响。之前的句子会作为种子文本、隐藏状态，参与到后续字符的生成。从下面的诗可以看出，“未死”、“不得”、“蹉跎”等意象是连续相近的。

​	发现韵脚不可避免地出现问题，并且前后句的字数也开始有不稳定的变化。或许可以认为是藏头字的强行加入扰乱了生成。

  以下采用了一些网友的网名。发现有一些有趣的现象。

1. 模型很喜欢“療”字，“何”字，“不”字等，后两个字认为应该是出现频率比较高，并且有比较固定的词语搭配，例如“何事”“不可”。但是“療”作为一个生僻字也经常出现。查询原始数据集，发现“療”字并不存在，也就是说这是词表中不存在的字。
2. 连续生成的诗会有一些相似性。比如前两首都以“不可惫”结尾，后两首的句式一模一样。但是在编写代码时，每生成一首诗都会重新初始化隐藏状态，不知道是为啥。

<img width="1536" height="435" alt="屏幕截图 2025-11-18 224937" src="https://github.com/user-attachments/assets/51bc2325-bd25-4a03-aca0-95d5271f9db3" />

<img width="1518" height="403" alt="屏幕截图 2025-11-18 225044" src="https://github.com/user-attachments/assets/969939cf-aacd-45c5-a0ab-d92d770a5315" />

<img width="1507" height="383" alt="屏幕截图 2025-11-18 225122" src="https://github.com/user-attachments/assets/2b8072d8-6263-4466-8423-fde091e2fcc3" />

<img width="1507" height="373" alt="屏幕截图 2025-11-18 225219" src="https://github.com/user-attachments/assets/b7606daf-5dd3-415e-9477-530612ae61b1" />



##### （3）超长诗

<img width="1218" height="1211" alt="屏幕截图 2025-11-18 225931" src="https://github.com/user-attachments/assets/184862ff-a75b-4b78-9565-523e797c6024" />

​	长诗和普通诗没有太大区别，但是我们依然可以发现以下特点。

1. 长诗的上下文很不通顺，跟模型记忆力有限有关。基本模型设计的隐藏层维度为256，其实是比较小的。并且在训练时，它考虑50个文本，更早的文本会被窗口遗忘。

2. 生成的句子比藏头诗更加规整，有连续的五言诗部分，夹杂一些古体和七言诗。从中也可以发现，在训练文本中，五言诗的占比是最大的，因此模型倾向于生成五言诗。

3. 在短的范围内，诗歌还是有一定的语义关联，例如下面这句：

   吏吟水绕元官神，得有圣作登虚空
   几时霜井故乡愁，身远应怜雪北隅。

   看起来前两句是官吏诗，后两句是羁旅诗。“吏”、“圣作”意象的关联比较明显。而下一句每一个意象都应了羁旅思乡的主题，是生成的诗歌中比较优秀的一句，猜测训练集中羁旅思乡诗歌数量比较多，因此模型学习效果比较好。

   甚至这句诗不仅是意象的堆砌，忽略掉糟糕的平仄用韵，甚至还比较能读通。

   <img width="1013" height="281" alt="屏幕截图 2025-11-19 154116" src="https://github.com/user-attachments/assets/3ae41768-606b-4b02-9673-6ba7ce48cdbb" />

#### 2、双向LSTM+自适应调度器

​	对BiLSTM模型的训练设置：batch_size=512，epoch=10，lr=0.001。吸取了之前lr太大的教训，可以看到收敛得还不错。然而，生成的诗歌质量比之前差很多，认为是严重过拟合。采用不同的checkpoint生成超长诗，观察结果：

<img width="1022" height="657" alt="cd6bed8127751e8fa3217eb07f5bc5f9" src="https://github.com/user-attachments/assets/6b2b60e3-f5eb-4cbd-8d44-0f6868537a27" />

##### （1）使用loss最低的模型（epoch10）

​	在生成超长诗时，开始生成一些古文，而不是古诗。句式、平仄、用韵杂乱。

<img width="1062" height="1106" alt="屏幕截图 2025-11-19 160106" src="https://github.com/user-attachments/assets/4325c4e5-24da-4dea-8f20-ec3a7f0122e4" />

##### （2）epoch_1

​	句式规整，学到了部分平仄，但是在句子中会重复出现一些字，说明学习得还不够。

<img width="273" height="761" alt="image" src="https://github.com/user-attachments/assets/8c42f5e2-1089-4b0c-a72c-0ebd1e0394f6" />

<img width="248" height="350" alt="image" src="https://github.com/user-attachments/assets/7e4b05c3-c9b5-4ae3-98cc-0c4ecd8ad744" />

##### （3）epoch_2

​	句式也比较规整，但是会开始有不同的句式，并且可以看到，生成了五言绝句，七言律诗，连续的五言诗和连续的七言诗。认为模型对训练集有了较好的学习。平仄表现较为优秀，但是出错也比较集中。

​	也有一些比较优秀的诗句，例如：不知频是乡愁恨，倍笑春寒不自催。海闲应不隔，同是有江东。

​	依旧存在重复用字的问题。例如：清杨叠石叠为烟，一夜轻云压倒车。江水流茫渺柳流，几回秋水夜寒沈。

<img width="362" height="788" alt="image" src="https://github.com/user-attachments/assets/b569ecfe-82cf-4e42-a45b-085e8166f348" />

<img width="869" height="284" alt="image" src="https://github.com/user-attachments/assets/6f07905d-3e13-4b6c-a880-0490287cda16" />

<img width="484" height="356" alt="image" src="https://github.com/user-attachments/assets/e2517ad1-4fde-4e15-bb79-8c0c745802a6" />

##### （4）epoch_3

​	句式开始有杂乱，疑似过拟合。开始刻意拟合到一些占比比较小的句式（或者词？）

​	但是也能发现一些比较有趣的句子，例如：春愁不得，长住无多。妙恨红泪，银丝新柳。日暮自相见，此是谁留待。感觉对语义的学习在加深，对格律的学习在变浅。

​	但是依旧不能解决重复字的问题。

<img width="411" height="763" alt="image" src="https://github.com/user-attachments/assets/225a4f39-d25a-4423-b519-44a97ffdb682" />

##### （5）epoch_4

​	句式开始变得规整。生成了一些楚辞感觉的东西，例如：金圣杞尊兮风物进，神朱盛德兮神风文。肃享歌兮庆天命，威枚澶兮虔候中。

​	在诗经体部分，生成的和君王、祭祀有关，例如：三太圣必，重临明祀。乐乱展享，管终和春。帝三王命，庶礼三良。广礼春毕，式和乐昌。看来学到了句式和语义之间的关联。

​	甚至发现偶尔出现了对仗，例如：玉翠寒红帐，旗幢满锦城。动词选用还挺灵性的。还有：凌霜彩霄杂，高步凤凰来。虽然不知道在说啥，但是有点像诗了。

<img width="572" height="729" alt="image" src="https://github.com/user-attachments/assets/5e2622ed-c19c-4597-8b54-0cc3e33f0b33" />

##### （6）别的一些有趣的句子摘录

​	epoch5：我本能以为主也，曾无高是天下后。我未得见谋，何以识名利。物矫以远，不得高如。长羽有哀，饥我不多。

​	epoch6：此时新有意，安得对黄天。上苑青云绕，玉峨振翠微。（这句有点感觉）陟风翠烟际，依触芳气中。无私指天子，谦让此龙庭。（很歌功颂德了）

​	epoch7：乞身尽是人情事，著得琴书称去难。（很有感觉）雪晚即云开，知吾自此来。（神仙下凡吗）眼中唯有酒，花酒伴正长。（有酒鬼）始待吴辰从别后，谁将文墨到重来。纵无不羡梦无情，便过因君求陆情。（句式很有感觉，但是内容像是梦话）

​	epoch8：弱凉醉无虞，牛马亦已枯。（牛马亦已哭）所为无儿言，免作出臣心。君闲思故人，不负家贫柳。（似乎学会用典了）

​	epoch9：终应白云远，便是入帘无。（对仗的梦话）庭尘更已近，来向浅无埃。

​	根据我的主观判断，我认为epoch3的时候有一点小小的过拟合，但是后来学到了细节，也没太跑偏。在epoch8的时候，开始又有过拟合的迹象了，虽然学到了一些神奇的遣词造句，但是比较梦话。到epoch10的时候严重过拟合，生成的质量下降得比较严重。

​	**但是总体来说，双向LSTM学到的比单向更好一些。**

#### 5、实验设计缺陷与未来优化

​	我感觉有以下缺陷：

1. 数据集处理不够干净，忘记把引号去除了，还有一些句子只有一个字，或者缺少标点，这种明显不是诗的东西也没有去除。

2. 给诗歌加标签很有意思，但是我没有训练出很好的权重，甚至后来加到30个epoch也生成了一些重复字的组合的东西。

3. 模型有时候喜欢使用重复字，并且在生成生僻字后会加大生成生僻字的概率，应该避免这一点。

   未来可以考虑调研文献，尝试更优化的算法。



## 六、实验结论

​	设计了自动写诗模型，能够通过参数控制生成普通诗歌、藏头诗和超长诗。

​	对比了单向LSTM和双向LSTM性能的差异，认为双向LSTM的性能更加优越。

​	感受了过大的学习率对于模型无法收敛的影响，了解了学习率调度器的工作原理。

​	通过主观感受，了解了诗歌生成模型过拟合的后果，认为在batch_size=512，lr=0.001，epoch在5-8之间可以生成比较像诗的诗。字数、平仄和韵脚是模型最容易理解的特征，语义相对困难一些，但是经过学习，模型能够学到常用词语与句式搭配，类似意象的叠用，甚至是对仗。


