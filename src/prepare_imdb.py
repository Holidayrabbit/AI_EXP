"""
IMDb 数据集预处理脚本
下载并处理 IMDb Large Movie Review Dataset
"""
import os
import re
import random
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def clean_text(text):
    """轻量级文本清洗"""
    # 去除 HTML 标签
    text = BeautifulSoup(text, "lxml").get_text()
    # 统一空白字符
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空白
    text = text.strip()
    return text

def prepare_imdb(data_dir='../data', seed=42):
    """
    准备 IMDb 数据集
    - 下载原始数据（使用 HuggingFace datasets）
    - 清洗文本
    - 划分训练集/验证集/测试集
    """
    random.seed(seed)
    
    # 创建目录
    raw_dir = Path(data_dir) / 'raw' / 'imdb'
    processed_dir = Path(data_dir) / 'processed' / 'imdb'
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在下载 IMDb 数据集...")
    # 使用 HuggingFace datasets 下载 IMDb
    dataset = load_dataset('imdb')
    
    # 处理训练集
    print("处理训练集...")
    train_data = []
    for item in dataset['train']:
        text = clean_text(item['text'])
        label = item['label']  # 0=neg, 1=pos
        train_data.append({'text': text, 'label': label})
    
    train_df = pd.DataFrame(train_data)
    
    # 划分训练集和验证集（90%-10%）
    train_df, valid_df = train_test_split(
        train_df, test_size=0.1, random_state=seed, stratify=train_df['label']
    )
    
    # 处理测试集
    print("处理测试集...")
    test_data = []
    for item in dataset['test']:
        text = clean_text(item['text'])
        label = item['label']
        test_data.append({'text': text, 'label': label})
    
    test_df = pd.DataFrame(test_data)
    
    # 保存处理后的数据
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    valid_df.to_csv(processed_dir / 'valid.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)
    
    print(f"IMDb 数据集处理完成:")
    print(f"  训练集: {len(train_df)} 条 (正样本: {(train_df['label']==1).sum()})")
    print(f"  验证集: {len(valid_df)} 条 (正样本: {(valid_df['label']==1).sum()})")
    print(f"  测试集: {len(test_df)} 条 (正样本: {(test_df['label']==1).sum()})")
    print(f"保存路径: {processed_dir}")
    
    return train_df, valid_df, test_df

if __name__ == '__main__':
    prepare_imdb()

