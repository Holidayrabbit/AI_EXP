#!/bin/bash
# 完整实验流程自动化脚本（CPU版本 - 适合MacBook Air）

set -e  # 遇到错误立即退出

echo "============================================"
echo "领域自适应情感分析实验 - CPU优化版"
echo "============================================"
echo "本脚本专为CPU环境优化，只运行传统机器学习模型"
echo "预计总时间：30-40分钟"
echo ""

# 进入 src 目录
cd src

# 步骤 1: 准备数据
echo "步骤 1: 准备 IMDb 数据集..."
python prepare_imdb.py

echo ""
echo "请确认 TripAdvisor 数据集已准备好："
echo "位置: ../data/processed/tripadvisor/test.csv"
echo ""
read -p "数据准备完成后按回车继续（按 Ctrl+C 退出）..."

# 步骤 2: 训练基线模型
echo ""
echo "============================================"
echo "步骤 2: 训练基线模型"
echo "============================================"

echo ""
echo "[1/7] 训练基线模型 - SVM（预计 5 分钟）..."
python train_traditional.py --method baseline --model svm

echo ""
echo "[2/7] 训练基线模型 - Naive Bayes（预计 3 分钟）..."
python train_traditional.py --method baseline --model nb

# 步骤 3: 训练领域自适应模型（合并训练）
echo ""
echo "============================================"
echo "步骤 3: 训练领域自适应模型（合并训练）"
echo "============================================"

echo ""
echo "[3/7] SVM + 目标域 1% 标注（预计 5 分钟）..."
python train_traditional.py --method combined --model svm --target_ratio 1pct

echo ""
echo "[4/7] SVM + 目标域 5% 标注（预计 6 分钟）..."
python train_traditional.py --method combined --model svm --target_ratio 5pct

echo ""
echo "[5/7] SVM + 目标域 10% 标注（预计 7 分钟）..."
python train_traditional.py --method combined --model svm --target_ratio 10pct

echo ""
echo "[6/7] NB + 目标域 5% 标注（预计 4 分钟）..."
python train_traditional.py --method combined --model nb --target_ratio 5pct

# 步骤 4: 生成评估报告
echo ""
echo "============================================"
echo "步骤 4: 生成评估报告"
echo "============================================"
echo ""
echo "[7/7] 生成混淆矩阵、性能对比图表..."
python evaluate.py

echo ""
echo "============================================"
echo "✓ 实验完成！"
echo "============================================"
echo ""
echo "结果保存在 ../results/ 目录："
echo "  - comparison.csv         性能对比表格"
echo "  - comparison.png         性能对比图"
echo "  - distribution.png       情感分布对比"
echo "  - confusion_matrix_*.png 混淆矩阵"
echo ""
echo "可以开始撰写实验报告了！"
echo ""

