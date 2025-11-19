import numpy as np
import torch
import pickle
import os

def sample(preds, temperature=1.0):
    """根据预测结果和温度生成下一个字符的索引"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

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

def generate_long_poem(model, vocab, seed_text=None, line_count=50, temperature=0.7, 
                      seq_length=50, device='cpu'):
    """生成超长诗（按句号换行，行数>=21）"""
    # 确保超长诗行数>=21
    line_count = max(line_count, 21)
    return generate_poem(model, vocab, seed_text, line_count, temperature, 
                        seq_length, device)

def load_vocab(vocab_path='preprocessed_data/vocab.pkl'):
    """加载词汇表"""
    with open(vocab_path, 'rb') as f:
        return pickle.load(f)