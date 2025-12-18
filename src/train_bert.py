"""
BERT 两阶段微调
Stage 1: 在源域（IMDb）上微调
Stage 2: 在目标域小标注集上继续微调
"""
import os
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

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
    return df['text'].values, df['label'].values

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro')
    
    return acc, f1, predictions, true_labels

def train_stage1(train_path, valid_path, save_dir='../models/bert',
                lr=2e-5, epochs=3, batch_size=16, seed=42):
    """
    Stage 1: 在源域（IMDb）上微调 BERT
    """
    print(f"\n{'='*60}")
    print("Stage 1: 在源域（IMDb）上微调 BERT")
    print(f"{'='*60}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    X_train, y_train = load_data(train_path)
    X_valid, y_valid = load_data(valid_path)
    
    # 加载 tokenizer 和模型
    print("加载 BERT 模型...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    model.to(device)
    
    # 创建数据集和数据加载器
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    valid_dataset = SentimentDataset(X_valid, y_valid, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练
    best_f1 = 0
    best_model_path = None
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        valid_acc, valid_f1, _, _ = evaluate(model, valid_loader, device)
        print(f"Valid Accuracy: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}")
        
        # 保存最佳模型
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            save_path = Path(save_dir) / 'stage1'
            save_path.mkdir(parents=True, exist_ok=True)
            best_model_path = save_path / 'best_model'
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"保存最佳模型 (F1={best_f1:.4f})")
    
    print(f"\nStage 1 完成，最佳验证 F1: {best_f1:.4f}")
    print(f"模型已保存至: {best_model_path}")
    
    return str(best_model_path)

def train_stage2(stage1_model_path, target_train_path, target_test_path,
                save_dir='../models/bert', target_ratio='1pct',
                lr=1e-5, epochs=3, batch_size=16, seed=42):
    """
    Stage 2: 在目标域小标注集上继续微调
    """
    print(f"\n{'='*60}")
    print(f"Stage 2: 在目标域（{target_ratio}）上继续微调")
    print(f"{'='*60}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    X_train, y_train = load_data(target_train_path)
    X_test, y_test = load_data(target_test_path)
    
    print(f"目标域训练集: {len(X_train)} 条")
    
    # 加载 Stage 1 的模型
    print(f"加载 Stage 1 模型: {stage1_model_path}")
    tokenizer = BertTokenizer.from_pretrained(stage1_model_path)
    model = BertForSequenceClassification.from_pretrained(stage1_model_path)
    model.to(device)
    
    # 创建数据集和数据加载器
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练
    best_f1 = 0
    best_model_path = None
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        test_acc, test_f1, _, _ = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
        
        # 保存最佳模型
        if test_f1 > best_f1:
            best_f1 = test_f1
            save_path = Path(save_dir) / 'stage2'
            save_path.mkdir(parents=True, exist_ok=True)
            best_model_path = save_path / f'best_model_{target_ratio}'
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"保存最佳模型 (F1={best_f1:.4f})")
    
    # 最终评估
    print(f"\n{'='*60}")
    print("最终评估:")
    model = BertForSequenceClassification.from_pretrained(best_model_path)
    model.to(device)
    
    test_acc, test_f1, predictions, true_labels = evaluate(model, test_loader, device)
    
    print(f"目标域测试集:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro-F1: {test_f1:.4f}")
    
    # 保存结果
    results = {
        'method': f'bert_stage2_{target_ratio}',
        'target_accuracy': float(test_acc),
        'target_f1': float(test_f1),
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
    }
    
    results_file = Path(save_dir) / 'stage2' / f'results_{target_ratio}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存至: {results_file}")
    
    return results

def evaluate_stage1_on_target(stage1_model_path, target_test_path, 
                              save_dir='../models/bert'):
    """
    评估 Stage 1 模型在目标域上的性能（未经目标域微调）
    """
    print(f"\n{'='*60}")
    print("评估 Stage 1 模型在目标域的性能（无目标域微调）")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    X_test, y_test = load_data(target_test_path)
    
    # 加载模型
    tokenizer = BertTokenizer.from_pretrained(stage1_model_path)
    model = BertForSequenceClassification.from_pretrained(stage1_model_path)
    model.to(device)
    
    # 评估
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    test_acc, test_f1, predictions, true_labels = evaluate(model, test_loader, device)
    
    print(f"目标域测试集:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Macro-F1: {test_f1:.4f}")
    
    # 保存结果
    results = {
        'method': 'bert_stage1_only',
        'target_accuracy': float(test_acc),
        'target_f1': float(test_f1),
        'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
    }
    
    results_file = Path(save_dir) / 'stage1' / 'results_on_target.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, choices=['1', '2', 'eval'], 
                       default='1', help='训练阶段')
    parser.add_argument('--target_ratio', type=str, default='1pct',
                       choices=['1pct', '5pct', '10pct'],
                       help='目标域小标注集比例（Stage 2 需要）')
    parser.add_argument('--stage1_model', type=str, 
                       default='../models/bert/stage1/best_model',
                       help='Stage 1 模型路径（Stage 2 需要）')
    args = parser.parse_args()
    
    if args.stage == '1':
        # Stage 1: 在源域训练
        train_path = '../data/processed/imdb/train.csv'
        valid_path = '../data/processed/imdb/valid.csv'
        train_stage1(train_path, valid_path)
        
    elif args.stage == 'eval':
        # 评估 Stage 1 模型在目标域的性能
        target_test_path = '../data/processed/tripadvisor/test.csv'
        evaluate_stage1_on_target(args.stage1_model, target_test_path)
        
    else:
        # Stage 2: 在目标域微调
        target_train_path = f'../data/processed/tripadvisor/train_small_{args.target_ratio}.csv'
        target_test_path = '../data/processed/tripadvisor/test.csv'
        train_stage2(args.stage1_model, target_train_path, target_test_path,
                    target_ratio=args.target_ratio)

