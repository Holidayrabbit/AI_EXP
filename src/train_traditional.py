"""
传统机器学习模型训练（SVM 和 Naive Bayes）
实现基线模型和领域自适应（合并训练）
"""
import os
import pickle
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json

def load_data(data_path):
    """加载数据"""
    df = pd.read_csv(data_path)
    return df['text'].values, df['label'].values

def train_baseline(source_train, source_valid, target_test, 
                  model_type='svm', save_dir='../models'):
    """
    基线模型：只在源域（IMDb）训练，在目标域（TripAdvisor）测试
    
    参数:
    - source_train: 源域训练集路径
    - source_valid: 源域验证集路径
    - target_test: 目标域测试集路径
    - model_type: 'svm' 或 'nb'
    """
    print(f"\n{'='*60}")
    print(f"训练基线模型: {model_type.upper()} (只用源域数据)")
    print(f"{'='*60}")
    
    # 加载数据
    X_train, y_train = load_data(source_train)
    X_valid, y_valid = load_data(source_valid)
    X_test_target, y_test_target = load_data(target_test)
    
    # TF-IDF 特征提取（只在源域训练集上 fit）
    print("提取 TF-IDF 特征...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=20000
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)
    X_test_target_tfidf = vectorizer.transform(X_test_target)
    
    print(f"特征维度: {X_train_tfidf.shape[1]}")
    
    # 训练模型（在验证集上调参）
    if model_type == 'svm':
        print("训练 Linear SVM...")
        best_score = 0
        best_model = None
        best_C = None
        
        for C in [0.1, 1.0, 10.0]:
            model = SGDClassifier(
                loss='hinge',
                alpha=1/(C*len(X_train)),  # alpha = 1/(C*n)
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
            model.fit(X_train_tfidf, y_train)
            valid_pred = model.predict(X_valid_tfidf)
            valid_f1 = f1_score(y_valid, valid_pred, average='macro')
            print(f"  C={C}: valid F1={valid_f1:.4f}")
            
            if valid_f1 > best_score:
                best_score = valid_f1
                best_model = model
                best_C = C
        
        print(f"最佳参数: C={best_C}, valid F1={best_score:.4f}")
        
    elif model_type == 'nb':
        print("训练 Naive Bayes...")
        best_score = 0
        best_model = None
        best_alpha = None
        
        for alpha in [0.5, 1.0, 2.0]:
            model = ComplementNB(alpha=alpha)
            model.fit(X_train_tfidf, y_train)
            valid_pred = model.predict(X_valid_tfidf)
            valid_f1 = f1_score(y_valid, valid_pred, average='macro')
            print(f"  alpha={alpha}: valid F1={valid_f1:.4f}")
            
            if valid_f1 > best_score:
                best_score = valid_f1
                best_model = model
                best_alpha = alpha
        
        print(f"最佳参数: alpha={best_alpha}, valid F1={best_score:.4f}")
    
    # 在目标域测试
    y_pred_target = best_model.predict(X_test_target_tfidf)
    acc = accuracy_score(y_test_target, y_pred_target)
    f1 = f1_score(y_test_target, y_pred_target, average='macro')
    
    print(f"\n目标域测试结果:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro-F1: {f1:.4f}")
    
    # 保存模型
    save_path = Path(save_dir) / 'traditional'
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_file = save_path / f'baseline_{model_type}.pkl'
    vectorizer_file = save_path / f'baseline_{model_type}_vectorizer.pkl'
    
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # 保存结果
    results = {
        'model_type': model_type,
        'method': 'baseline',
        'target_accuracy': float(acc),
        'target_f1': float(f1),
        'confusion_matrix': confusion_matrix(y_test_target, y_pred_target).tolist()
    }
    
    results_file = save_path / f'baseline_{model_type}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"模型已保存至: {model_file}")
    
    return results

def train_combined(source_train, source_valid, target_train_small, target_test,
                   model_type='svm', save_dir='../models', small_ratio='1pct'):
    """
    领域自适应：合并源域和目标域小标注集进行训练
    """
    print(f"\n{'='*60}")
    print(f"训练合并模型: {model_type.upper()} (源域 + 目标域 {small_ratio})")
    print(f"{'='*60}")
    
    # 加载数据
    X_train_src, y_train_src = load_data(source_train)
    X_valid, y_valid = load_data(source_valid)
    X_train_tgt, y_train_tgt = load_data(target_train_small)
    X_test_target, y_test_target = load_data(target_test)
    
    # 合并训练数据
    X_train = np.concatenate([X_train_src, X_train_tgt])
    y_train = np.concatenate([y_train_src, y_train_tgt])
    
    print(f"合并训练集: {len(X_train)} 条 (源域: {len(X_train_src)}, 目标域: {len(X_train_tgt)})")
    
    # TF-IDF 特征提取（在合并数据上 fit）
    print("提取 TF-IDF 特征...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=20000
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)
    X_test_target_tfidf = vectorizer.transform(X_test_target)
    
    # 训练模型
    if model_type == 'svm':
        print("训练 Linear SVM...")
        best_score = 0
        best_model = None
        best_C = None
        
        for C in [0.1, 1.0, 10.0]:
            model = SGDClassifier(
                loss='hinge',
                alpha=1/(C*len(X_train)),
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
            model.fit(X_train_tfidf, y_train)
            valid_pred = model.predict(X_valid_tfidf)
            valid_f1 = f1_score(y_valid, valid_pred, average='macro')
            print(f"  C={C}: valid F1={valid_f1:.4f}")
            
            if valid_f1 > best_score:
                best_score = valid_f1
                best_model = model
                best_C = C
        
        print(f"最佳参数: C={best_C}")
        
    elif model_type == 'nb':
        print("训练 Naive Bayes...")
        best_score = 0
        best_model = None
        best_alpha = None
        
        for alpha in [0.5, 1.0, 2.0]:
            model = ComplementNB(alpha=alpha)
            model.fit(X_train_tfidf, y_train)
            valid_pred = model.predict(X_valid_tfidf)
            valid_f1 = f1_score(y_valid, valid_pred, average='macro')
            print(f"  alpha={alpha}: valid F1={valid_f1:.4f}")
            
            if valid_f1 > best_score:
                best_score = valid_f1
                best_model = model
                best_alpha = alpha
        
        print(f"最佳参数: alpha={best_alpha}")
    
    # 在目标域测试
    y_pred_target = best_model.predict(X_test_target_tfidf)
    acc = accuracy_score(y_test_target, y_pred_target)
    f1 = f1_score(y_test_target, y_pred_target, average='macro')
    
    print(f"\n目标域测试结果:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro-F1: {f1:.4f}")
    
    # 保存模型
    save_path = Path(save_dir) / 'traditional'
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_file = save_path / f'combined_{model_type}_{small_ratio}.pkl'
    vectorizer_file = save_path / f'combined_{model_type}_{small_ratio}_vectorizer.pkl'
    
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # 保存结果
    results = {
        'model_type': model_type,
        'method': f'combined_{small_ratio}',
        'target_accuracy': float(acc),
        'target_f1': float(f1),
        'confusion_matrix': confusion_matrix(y_test_target, y_pred_target).tolist()
    }
    
    results_file = save_path / f'combined_{model_type}_{small_ratio}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"模型已保存至: {model_file}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['baseline', 'combined'], 
                       default='baseline', help='训练方法')
    parser.add_argument('--model', type=str, choices=['svm', 'nb'], 
                       default='svm', help='模型类型')
    parser.add_argument('--target_ratio', type=str, default='1pct',
                       choices=['1pct', '5pct', '10pct'],
                       help='目标域小标注集比例（仅 combined 方法需要）')
    args = parser.parse_args()
    
    # 数据路径
    source_train = '../data/processed/imdb/train.csv'
    source_valid = '../data/processed/imdb/valid.csv'
    target_test = '../data/processed/tripadvisor/test.csv'
    
    if args.method == 'baseline':
        train_baseline(source_train, source_valid, target_test, 
                      model_type=args.model)
    else:
        target_train_small = f'../data/processed/tripadvisor/train_small_{args.target_ratio}.csv'
        train_combined(source_train, source_valid, target_train_small, target_test,
                      model_type=args.model, small_ratio=args.target_ratio)

