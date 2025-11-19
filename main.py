import argparse
import torch
from generate import (
    generate_poem, generate_acrostic_poem, generate_long_poem, load_vocab
)
from model import PoetryRNN, load_model

"""
python main.py --type normal --length 4 --model_path Bimodels/best_model.pth
python main.py --type acrostic --head_chars "我自风来" --lines_per_head 1 --temperature 0.6 
python main.py --type long --length 30 --gpu_id 6 --temperature 0.8
"""


def main():
    parser = argparse.ArgumentParser(description='写诗机器人 (PyTorch版本)')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='模型路径')
    parser.add_argument('--vocab_path', type=str, default='preprocessed_data/vocab.pkl', help='词汇表路径')
    parser.add_argument('--type', type=str, default='normal', choices=['normal', 'acrostic', 'long'], help='诗歌类型')
    parser.add_argument('--length', type=int, default=20, help='诗歌行数（普通诗/超长诗用）')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度，值越高越随机')
    parser.add_argument('--seq_length', type=int, default=50, help='序列长度，需与训练时一致')
    parser.add_argument('--embed_dim', type=int, default=128, help='嵌入维度，需与训练时一致')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM隐藏层维度，需与训练时一致')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数，需与训练时一致')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout率，需与训练时一致')
    parser.add_argument('--head_chars', type=str, default='春夏秋冬', help='藏头字符，用于藏头诗')
    # 新增：每个藏头字对应的行数（适配藏头诗函数）
    parser.add_argument('--lines_per_head', type=int, default=1, help='每个藏头字对应的诗句数量（藏头诗用）')
    # 新增：指定GPU编号（默认0号GPU，和训练时一致）
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU编号')
    
    args = parser.parse_args()
    
    # 设备设置：使用指定编号的GPU
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载词汇表
    print("加载词汇表...")
    vocab = load_vocab(args.vocab_path)
    vocab_size = vocab['vocab_size']
    
    # 初始化并加载模型
    print("加载模型...")
    model = PoetryRNN(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    model = load_model(model, args.model_path)
    model.eval()
    
    # 生成诗歌
    print("\n生成诗歌中...\n")
    if args.type == 'normal':
        poem = generate_poem(
            model, vocab, 
            line_count=args.length,  # 注意：参数名改为line_count（适配修改后的generate_poem）
            temperature=args.temperature,
            seq_length=args.seq_length,
            device=device
        )
    elif args.type == 'acrostic':
        poem = generate_acrostic_poem(
            model, vocab,
            head_chars=args.head_chars,
            lines_per_head=args.lines_per_head,  # 传递新增参数
            temperature=args.temperature,
            seq_length=args.seq_length,
            device=device
        )
    else:  # long
        # 确保超长诗大于20行
        line_count = max(args.length, 21)
        poem = generate_long_poem(
            model, vocab,
            line_count=line_count,  # 参数名改为line_count
            temperature=args.temperature,
            seq_length=args.seq_length,
            device=device
        )
    
    # 显示结果
    print("生成的诗歌：")
    print("-" * 50)
    print(poem)
    print("-" * 50)

if __name__ == "__main__":
    main()