import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
import os
import pickle
from data_preprocessing import (
    load_and_preprocess_data, 
    create_vocabulary, 
    get_data_loader
)
from model import PoetryBiRNN, save_model, load_model, PositionalEncoding

"""
python train.py --data_path data/raw_poems.txt --epochs 10 --batch_size 512 --lr 0.001 --resume //--hidden_dim 512
"""


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    for sequences, targets in tqdm(train_loader, desc="Training"):
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(sequences) # 
        outputs = outputs[:, -1, :] #  
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def main():
    parser = argparse.ArgumentParser(description='训练写诗机器人模型 (带学习率调度器+每epoch存权重)')
    parser.add_argument('--data_path', type=str, required=True, help='训练数据文件路径')
    parser.add_argument('--seq_length', type=int, default=50, help='序列长度')
    parser.add_argument('--epochs', type=int, default=50, help='总训练轮数')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--embed_dim', type=int, default=128, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout率')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--model_dir', type=str, default='Bimodels', help='模型保存目录')
    parser.add_argument('--resume', action='store_true', help='是否从断点续训')
    # 新增：是否只保留最新的N个epoch权重（避免磁盘占用过大）
    parser.add_argument('--keep_last', type=int, default=10, help='保留最新的epoch权重数量（默认10个）')
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建epoch权重保存子目录（避免和best/final模型混淆）
    epoch_model_dir = os.path.join(args.model_dir, 'epoch_weights')
    if not os.path.exists(epoch_model_dir):
        os.makedirs(epoch_model_dir)
    
    # 数据准备
    print("加载并预处理数据...")
    text = load_and_preprocess_data(args.data_path)
    vocab = create_vocabulary(text) # 制作词表
    vocab_size = vocab['vocab_size'] # 词表大小
    print(f"词汇表大小: {vocab_size}")
    
    train_loader = get_data_loader(
        text, 
        vocab, 
        seq_length=args.seq_length, 
        batch_size=args.batch_size
    )
    
    # 模型、损失函数、优化器初始化
    # model = PoetryRNN(
    #     vocab_size=vocab_size,
    #     embed_dim=args.embed_dim,
    #     hidden_dim=args.hidden_dim,
    #     num_layers=args.num_layers,
    #     dropout=args.dropout
    # ).to(device)

    model = PoetryBiRNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)


    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # # 初始化学习率调度器（去掉verbose参数，适配低版本PyTorch）
    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',          # 针对损失（越小越好）
    #     factor=0.5,          # 学习率调整倍数（乘以0.5）
    #     patience=3,          # 3轮损失不下降则调整
    #     min_lr=1e-6          # 最小学习率（避免过低）
    # )

    # 定义指数衰减函数
    def exponential_decay(epoch):
        decay_rate = 0.9    # 衰减率（如0.9表示每步衰减10%）
        decay_steps = 1    # 衰减步数（每1个epoch衰减一次）
        return decay_rate **(epoch / decay_steps)

    # 初始化调度器
    scheduler = LambdaLR(optimizer, lr_lambda=exponential_decay)

        
    # 断点续训相关变量
    start_epoch = 0
    best_loss = float('inf')
    prev_lr = optimizer.param_groups[0]['lr']  # 记录上一轮学习率
    
    # 检查是否需要续训
    if args.resume:
        state_path = os.path.join(args.model_dir, 'training_state.pkl')
        # 续训时优先加载最近的epoch权重（若存在），否则加载best_model
        latest_epoch_model = None
        if os.path.exists(epoch_model_dir):
            # 获取所有epoch权重文件，按epoch编号排序
            epoch_files = [f for f in os.listdir(epoch_model_dir) if f.startswith('epoch_') and f.endswith('.pth')]
            if epoch_files:
                epoch_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
                latest_epoch_model = os.path.join(epoch_model_dir, epoch_files[0])
        
        model_path = latest_epoch_model if latest_epoch_model else os.path.join(args.model_dir, 'best_model.pth')
        
        if os.path.exists(state_path) and os.path.exists(model_path):
            # 加载模型参数（优先最近的epoch权重）
            model = load_model(model, model_path)
            # 加载训练状态
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                start_epoch = state['epoch']
                best_loss = state['best_loss']
                optimizer.load_state_dict(state['optimizer_state'])
                scheduler.load_state_dict(state['scheduler_state'])
            
            # 续训时手动更新学习率（若指定）
            if args.lr is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                print(f"续训时手动更新学习率为: {args.lr:.6f}")
            
            prev_lr = optimizer.param_groups[0]['lr']
            print(f"从断点续训，起始轮数: {start_epoch + 1}, 最佳损失: {best_loss:.4f}")
            print(f"加载的模型权重: {os.path.basename(model_path)}")
            print(f"当前学习率: {prev_lr:.6f}")
        else:
            print("未找到断点文件，将从头开始训练")
    
    # 训练循环
    print("开始训练模型...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印当前轮次信息（包含学习率）
        print(f"Epoch {epoch+1}/{args.epochs}, 损失: {train_loss:.4f}, 学习率: {current_lr:.6f}")
        
        # 调度器根据损失调整学习率
        scheduler.step() # 调度器直接调节学习率
        # scheduler.step(train_loss)
        
        # 手动检测学习率是否变化，打印调整信息
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != prev_lr:
            print(f"学习率调整: {prev_lr:.6f} → {new_lr:.6f}")
            prev_lr = new_lr
        
        # 关键修改1：每个epoch保存当前轮次的权重
        epoch_model_path = os.path.join(epoch_model_dir, f'epoch_{epoch+1}.pth')
        save_model(model, epoch_model_path)
        print(f"保存第 {epoch+1} 轮权重到 {epoch_model_path}")
        
        # 关键修改2：清理过旧的epoch权重（只保留最新的N个）
        if args.keep_last > 0:
            epoch_files = [f for f in os.listdir(epoch_model_dir) if f.startswith('epoch_') and f.endswith('.pth')]
            if len(epoch_files) > args.keep_last:
                # 按epoch编号升序排序，删除最旧的
                epoch_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                for old_file in epoch_files[:-args.keep_last]:
                    os.remove(os.path.join(epoch_model_dir, old_file))
                    print(f"删除过旧权重文件: {old_file}")
        
        # 关键修改：每个epoch都保存训练状态（包含当前轮数），而非仅在最佳模型时保存
        state_path = os.path.join(args.model_dir, 'training_state.pkl')
        state = {
            'epoch': epoch,  # 记录当前轮数（无论是否最佳）
            'best_loss': best_loss,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

        # 保留原有逻辑：保存最佳模型和训练状态（用于续训和最佳效果）
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = os.path.join(args.model_dir, 'best_model.pth')
            save_model(model, best_model_path)
            
            print(f"更新并保存最佳模型到 {best_model_path}，训练状态已保存")
    
    # 保存最终模型（训练结束后的最终状态）
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    save_model(model, final_model_path)
    print(f"训练完成，最终模型保存到 {final_model_path}")
    print(f"所有epoch权重保存在: {epoch_model_dir}（仅保留最新{args.keep_last}个）")

if __name__ == "__main__":
    main()