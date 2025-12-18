#!/bin/bash
# 完整实验流程自动化脚本

set -e  # 遇到错误立即退出

echo "============================================"
echo "领域自适应情感分析实验 - 完整流程"
echo "============================================"

# 进入 src 目录
cd src

# 步骤 1: 准备数据
echo ""
echo "步骤 1: 准备 IMDb 数据集..."
python prepare_imdb.py

echo ""
echo "请手动准备 TripAdvisor 数据集："
echo "1. 下载数据集到 ../data/raw/ 目录"
echo "2. 运行: python prepare_tripadvisor.py ../data/raw/tripadvisor_reviews.csv"
echo ""
read -p "数据准备完成后按回车继续..."

# 步骤 2: 训练基线模型
echo ""
echo "步骤 2: 训练基线模型（SVM）..."
python train_traditional.py --method baseline --model svm

echo ""
echo "步骤 2: 训练基线模型（NB）..."
python train_traditional.py --method baseline --model nb

# 步骤 3: 训练合并模型
echo ""
echo "步骤 3: 训练合并模型（SVM + 1%）..."
python train_traditional.py --method combined --model svm --target_ratio 1pct

echo ""
echo "步骤 3: 训练合并模型（SVM + 5%）..."
python train_traditional.py --method combined --model svm --target_ratio 5pct

echo ""
echo "步骤 3: 训练合并模型（SVM + 10%）..."
python train_traditional.py --method combined --model svm --target_ratio 10pct

# 步骤 4: BERT Stage 1
echo ""
echo "步骤 4: BERT Stage 1 (在源域微调)..."
python train_bert.py --stage 1

# 步骤 5: 评估 BERT Stage 1
echo ""
echo "步骤 5: 评估 BERT Stage 1 在目标域的性能..."
python train_bert.py --stage eval

# 步骤 6: BERT Stage 2
echo ""
echo "步骤 6: BERT Stage 2 (目标域 1% 微调)..."
python train_bert.py --stage 2 --target_ratio 1pct

echo ""
echo "步骤 6: BERT Stage 2 (目标域 5% 微调)..."
python train_bert.py --stage 2 --target_ratio 5pct

echo ""
echo "步骤 6: BERT Stage 2 (目标域 10% 微调)..."
python train_bert.py --stage 2 --target_ratio 10pct

# 步骤 7: 生成评估报告
echo ""
echo "步骤 7: 生成评估报告..."
python evaluate.py

echo ""
echo "============================================"
echo "实验完成！"
echo "============================================"
echo "结果保存在 ../results/ 目录"

