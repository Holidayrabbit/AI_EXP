"""
统一评估脚本
- 生成混淆矩阵
- 分析常见错误类型
- 对比不同方法的性能
- 可视化结果
"""
import os
import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SentimentDataset(Dataset):
    """情感分析数据集"""
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_data(data_path):
    """加载数据"""
    df = pd.read_csv(data_path)
    return df

def evaluate_traditional_model(model_path, vectorizer_path, test_data):
    """评估传统模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    X_test = vectorizer.transform(test_data['text'].values)
    y_pred = model.predict(X_test)
    
    return y_pred

def evaluate_bert_model(model_path, test_data):
    """评估 BERT 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    dataset = SentimentDataset(
        test_data['text'].values, 
        test_data['label'].values, 
        tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=16)
    
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['负面', '正面'],
                yticklabels=['负面', '正面'])
    plt.title(title)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵已保存: {save_path}")

def analyze_errors(test_data, y_pred, n_samples=20):
    """分析常见错误类型"""
    test_data = test_data.copy()
    test_data['prediction'] = y_pred
    test_data['correct'] = test_data['label'] == test_data['prediction']
    
    errors = test_data[~test_data['correct']]
    
    print(f"\n错误样本分析 (共 {len(errors)} 条错误):")
    print(f"  假阳性 (预测正面，实际负面): {((y_pred == 1) & (test_data['label'] == 0)).sum()}")
    print(f"  假阴性 (预测负面，实际正面): {((y_pred == 0) & (test_data['label'] == 1)).sum()}")
    
    # 随机抽样错误样本
    if len(errors) > 0:
        sample_errors = errors.sample(min(n_samples, len(errors)), random_state=42)
        
        print(f"\n抽样 {len(sample_errors)} 条错误样本:")
        for idx, row in sample_errors.iterrows():
            label_name = '正面' if row['label'] == 1 else '负面'
            pred_name = '正面' if row['prediction'] == 1 else '负面'
            text_preview = row['text'][:100] + '...' if len(row['text']) > 100 else row['text']
            print(f"\n[真实: {label_name}, 预测: {pred_name}]")
            print(f"{text_preview}")
        
        return sample_errors
    
    return None

def compare_methods(results_dir='../models', save_dir='../results'):
    """对比不同方法的性能"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有结果
    all_results = []
    
    # 传统模型结果
    traditional_dir = Path(results_dir) / 'traditional'
    if traditional_dir.exists():
        for result_file in traditional_dir.glob('*_results.json'):
            with open(result_file, 'r') as f:
                result = json.load(f)
                all_results.append(result)
    
    # BERT 结果
    bert_dir = Path(results_dir) / 'bert'
    
    # Stage 1 only
    stage1_result = bert_dir / 'stage1' / 'results_on_target.json'
    if stage1_result.exists():
        with open(stage1_result, 'r') as f:
            all_results.append(json.load(f))
    
    # Stage 2
    stage2_dir = bert_dir / 'stage2'
    if stage2_dir.exists():
        for result_file in stage2_dir.glob('results_*.json'):
            with open(result_file, 'r') as f:
                all_results.append(json.load(f))
    
    # 创建对比表格
    if len(all_results) > 0:
        df_results = pd.DataFrame(all_results)
        df_results = df_results[['method', 'target_accuracy', 'target_f1']]
        df_results = df_results.sort_values('target_f1', ascending=False)
        
        print("\n" + "="*60)
        print("方法对比 (目标域测试集)")
        print("="*60)
        print(df_results.to_string(index=False))
        
        # 保存表格
        df_results.to_csv(save_path / 'comparison.csv', index=False)
        
        # 绘制对比图
        plt.figure(figsize=(12, 6))
        
        x = range(len(df_results))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], df_results['target_accuracy'], 
                width, label='Accuracy', alpha=0.8)
        plt.bar([i + width/2 for i in x], df_results['target_f1'], 
                width, label='Macro-F1', alpha=0.8)
        
        plt.xlabel('方法')
        plt.ylabel('分数')
        plt.title('不同方法在目标域的性能对比')
        plt.xticks(x, df_results['method'], rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / 'comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n对比结果已保存至: {save_path}")

def analyze_distribution(source_data_path, target_data_path, save_dir='../results'):
    """分析源域和目标域的情感分布"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    source_df = pd.read_csv(source_data_path)
    target_df = pd.read_csv(target_data_path)
    
    source_pos_ratio = (source_df['label'] == 1).sum() / len(source_df)
    target_pos_ratio = (target_df['label'] == 1).sum() / len(target_df)
    
    print("\n" + "="*60)
    print("情感分布对比")
    print("="*60)
    print(f"源域 (IMDb):")
    print(f"  正面: {(source_df['label']==1).sum()} ({source_pos_ratio:.2%})")
    print(f"  负面: {(source_df['label']==0).sum()} ({1-source_pos_ratio:.2%})")
    print(f"\n目标域 (TripAdvisor):")
    print(f"  正面: {(target_df['label']==1).sum()} ({target_pos_ratio:.2%})")
    print(f"  负面: {(target_df['label']==0).sum()} ({1-target_pos_ratio:.2%})")
    
    # 绘制分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 源域
    source_counts = source_df['label'].value_counts().sort_index()
    ax1.bar(['负面', '正面'], source_counts.values, color=['#ff6b6b', '#51cf66'])
    ax1.set_title('源域 (IMDb) 情感分布')
    ax1.set_ylabel('样本数量')
    for i, v in enumerate(source_counts.values):
        ax1.text(i, v + 100, str(v), ha='center', va='bottom')
    
    # 目标域
    target_counts = target_df['label'].value_counts().sort_index()
    ax2.bar(['负面', '正面'], target_counts.values, color=['#ff6b6b', '#51cf66'])
    ax2.set_title('目标域 (TripAdvisor) 情感分布')
    ax2.set_ylabel('样本数量')
    for i, v in enumerate(target_counts.values):
        ax2.text(i, v + max(target_counts.values)*0.02, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path / 'distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n分布图已保存: {save_path / 'distribution.png'}")

def main():
    """主函数：生成完整的评估报告"""
    print("="*60)
    print("生成评估报告")
    print("="*60)
    
    # 1. 分析情感分布
    print("\n步骤 1: 分析情感分布...")
    source_train = '../data/processed/imdb/train.csv'
    target_test = '../data/processed/tripadvisor/test.csv'
    
    if os.path.exists(source_train) and os.path.exists(target_test):
        analyze_distribution(source_train, target_test)
    
    # 2. 对比不同方法
    print("\n步骤 2: 对比不同方法...")
    compare_methods()
    
    # 3. 生成混淆矩阵（如果有预测结果）
    print("\n步骤 3: 生成混淆矩阵...")
    
    # 加载测试数据
    test_data = load_data(target_test)
    results_dir = Path('../results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 评估基线 SVM
    svm_model = '../models/traditional/baseline_svm.pkl'
    svm_vectorizer = '../models/traditional/baseline_svm_vectorizer.pkl'
    if os.path.exists(svm_model) and os.path.exists(svm_vectorizer):
        print("\n评估基线 SVM...")
        y_pred = evaluate_traditional_model(svm_model, svm_vectorizer, test_data)
        plot_confusion_matrix(
            test_data['label'].values, y_pred,
            '基线 SVM 混淆矩阵 (目标域)',
            results_dir / 'confusion_matrix_baseline_svm.png'
        )
        analyze_errors(test_data, y_pred)
    
    # 评估 BERT Stage 1
    bert_stage1 = '../models/bert/stage1/best_model'
    if os.path.exists(bert_stage1):
        print("\n评估 BERT Stage 1 (仅源域训练)...")
        y_pred = evaluate_bert_model(bert_stage1, test_data)
        plot_confusion_matrix(
            test_data['label'].values, y_pred,
            'BERT Stage 1 混淆矩阵 (目标域)',
            results_dir / 'confusion_matrix_bert_stage1.png'
        )
        analyze_errors(test_data, y_pred)
    
    # 评估 BERT Stage 2
    for ratio in ['1pct', '5pct', '10pct']:
        bert_stage2 = f'../models/bert/stage2/best_model_{ratio}'
        if os.path.exists(bert_stage2):
            print(f"\n评估 BERT Stage 2 ({ratio})...")
            y_pred = evaluate_bert_model(bert_stage2, test_data)
            plot_confusion_matrix(
                test_data['label'].values, y_pred,
                f'BERT Stage 2 ({ratio}) 混淆矩阵 (目标域)',
                results_dir / f'confusion_matrix_bert_stage2_{ratio}.png'
            )
            analyze_errors(test_data, y_pred)
    
    print("\n" + "="*60)
    print("评估报告生成完成！")
    print("="*60)
    print(f"结果保存在: {results_dir}")

if __name__ == '__main__':
    main()

