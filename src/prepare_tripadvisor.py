"""
TripAdvisor 数据集预处理脚本
需要先手动下载数据集，然后运行此脚本进行处理
"""
import os
import re
import random
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    """轻量级文本清洗"""
    if pd.isna(text):
        return ""
    # 去除 HTML 标签
    text = BeautifulSoup(str(text), "lxml").get_text()
    # 统一空白字符
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空白
    text = text.strip()
    return text

def map_rating_to_label(rating):
    """
    将评分映射为二分类标签
    1-2 星 -> 0 (负面)
    4-5 星 -> 1 (正面)
    3 星 -> 丢弃（减少噪声）
    """
    if rating in [1, 2]:
        return 0
    elif rating in [4, 5]:
        return 1
    else:
        return -1  # 3星标记为-1，后续过滤

def prepare_tripadvisor(raw_file, data_dir='../data', 
                        sample_ratios=[0.01, 0.05, 0.10], seed=42):
    """
    准备 TripAdvisor 数据集
    
    参数:
    - raw_file: 原始 CSV 文件路径（需包含 'Review'/'Review_Text' 和 'Rating' 列）
    - data_dir: 数据目录
    - sample_ratios: 小标注集的采样比例列表
    - seed: 随机种子
    """
    random.seed(seed)
    
    # 创建目录
    processed_dir = Path(data_dir) / 'processed' / 'tripadvisor'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("读取原始数据...")
    df = pd.read_csv(raw_file)
    
    # 检测文本列名（可能是 Review, Review_Text, review, text 等）
    text_col = None
    for col in ['Review', 'Review_Text', 'review', 'text', 'review_text']:
        if col in df.columns:
            text_col = col
            break
    
    # 检测评分列名
    rating_col = None
    for col in ['Rating', 'rating', 'score']:
        if col in df.columns:
            rating_col = col
            break
    
    if text_col is None or rating_col is None:
        print(f"错误：未找到文本列或评分列")
        print(f"当前列名: {df.columns.tolist()}")
        return
    
    print(f"使用列: 文本={text_col}, 评分={rating_col}")
    
    # 清洗文本和映射标签
    print("清洗文本并映射标签...")
    df['text'] = df[text_col].apply(clean_text)
    df['label'] = df[rating_col].apply(map_rating_to_label)
    
    # 过滤掉3星评论和空文本
    df = df[df['label'] != -1]
    df = df[df['text'].str.len() > 0]
    df = df[['text', 'label']].reset_index(drop=True)
    
    print(f"过滤后剩余 {len(df)} 条数据")
    print(f"  正样本: {(df['label']==1).sum()}")
    print(f"  负样本: {(df['label']==0).sum()}")
    
    # 划分测试集（20%）和 pool（80%）
    pool_df, test_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df['label']
    )
    
    # 保存测试集
    test_df.to_csv(processed_dir / 'test.csv', index=False)
    print(f"测试集: {len(test_df)} 条")
    
    # 从 pool 中抽取小标注集
    for ratio in sample_ratios:
        train_small_df, _ = train_test_split(
            pool_df, train_size=ratio, random_state=seed, 
            stratify=pool_df['label']
        )
        filename = f'train_small_{int(ratio*100)}pct.csv'
        train_small_df.to_csv(processed_dir / filename, index=False)
        print(f"小标注集 ({int(ratio*100)}%): {len(train_small_df)} 条")
    
    # 保存完整 pool（可用于无监督学习）
    pool_df.to_csv(processed_dir / 'pool.csv', index=False)
    
    print(f"TripAdvisor 数据集处理完成，保存路径: {processed_dir}")
    
    return test_df

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python prepare_tripadvisor.py <原始CSV文件路径>")
        print("示例: python prepare_tripadvisor.py ../data/raw/tripadvisor_hotel_reviews.csv")
    else:
        raw_file = sys.argv[1]
        if not os.path.exists(raw_file):
            print(f"错误：文件不存在 {raw_file}")
        else:
            prepare_tripadvisor(raw_file)

