# 快速开始指南（CPU 优化版）

> **💻 专为 MacBook Air 等 CPU 环境优化**  
> 无需 GPU，30-40 分钟完成全部实验，性能优秀（75-82% 准确率）

## ⚡ 5 分钟快速上手

### 1. 安装依赖（2 分钟）

```bash
cd /Users/zq/work/course/AI_EXP
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备数据（3 分钟）

```bash
cd src

# IMDb 自动下载
python prepare_imdb.py

# TripAdvisor 需要先下载（见 DATA_GUIDE.md）
# 下载后运行：
python prepare_tripadvisor.py ../data/raw/tripadvisor_reviews.csv
```

### 3. 运行实验

**方案 A：一键运行完整实验（推荐，30-40 分钟）⭐**

```bash
cd /Users/zq/work/course/AI_EXP
./run_all.sh
```

运行内容：
- ✅ 基线模型（SVM + NB）
- ✅ 领域自适应（3种数据比例）
- ✅ 自动生成评估报告
- ✅ **全程 CPU 运行，MacBook Air 完美支持**

**方案 B：分步运行（理解每个步骤）**

```bash
cd src

# 基线模型（8分钟）
python train_traditional.py --method baseline --model svm
python train_traditional.py --method baseline --model nb

# 领域自适应（18分钟）
python train_traditional.py --method combined --model svm --target_ratio 1pct
python train_traditional.py --method combined --model svm --target_ratio 5pct
python train_traditional.py --method combined --model svm --target_ratio 10pct

# 生成报告（2分钟）
python evaluate.py
```

---

## 📊 实验检查清单

- [ ] IMDb 数据已下载并预处理（3 个 CSV 文件）
- [ ] TripAdvisor 数据已下载并预处理（4 个 CSV 文件）
- [ ] 至少训练 1 个基线模型（SVM baseline）
- [ ] 至少训练 1 个自适应模型（SVM combined 或 BERT stage2）
- [ ] 运行 evaluate.py 生成报告
- [ ] 查看 results/ 目录的图表和表格

---

## 📁 数据文件检查

运行以下命令检查数据是否准备好：

```bash
ls data/processed/imdb/
# 应该看到: train.csv  valid.csv  test.csv

ls data/processed/tripadvisor/
# 应该看到: test.csv  train_small_1pct.csv  train_small_5pct.csv  train_small_10pct.csv  pool.csv
```

---

## 🎯 最小实验方案（15 分钟）⚡

如果时间极度紧张，最小可用方案：

```bash
cd src

# 1. 数据准备（如已准备好可跳过）
python prepare_imdb.py
python prepare_tripadvisor.py ../data/raw/tripadvisor_reviews.csv

# 2. 基线 SVM（5 分钟）
python train_traditional.py --method baseline --model svm

# 3. 自适应 SVM 5%（6 分钟）
python train_traditional.py --method combined --model svm --target_ratio 5pct

# 4. 生成报告（2 分钟）
python evaluate.py
```

**这个方案已足以**：
- ✅ 展示跨域性能下降（基线）
- ✅ 展示领域自适应效果（合并训练）
- ✅ 获得完整的评估报告和图表

---

## 🚀 推荐实验方案（30-40 分钟）⭐

完整但高效的 CPU 方案：

```bash
cd src

# 1. 数据准备（10 分钟）
python prepare_imdb.py
python prepare_tripadvisor.py ../data/raw/tripadvisor_reviews.csv

# 2. 基线模型（8 分钟）
python train_traditional.py --method baseline --model svm
python train_traditional.py --method baseline --model nb

# 3. 领域自适应 - 多个数据比例（18 分钟）
python train_traditional.py --method combined --model svm --target_ratio 1pct
python train_traditional.py --method combined --model svm --target_ratio 5pct
python train_traditional.py --method combined --model svm --target_ratio 10pct
python train_traditional.py --method combined --model nb --target_ratio 5pct

# 4. 生成完整报告（2 分钟）
python evaluate.py
```

**或直接运行**：
```bash
cd /Users/zq/work/course/AI_EXP
./run_all.sh
```

---

## 📈 结果查看

运行完成后，查看结果：

```bash
# 查看性能对比
cat ../results/comparison.csv

# 查看图表
open ../results/comparison.png
open ../results/distribution.png
open ../results/confusion_matrix_baseline_svm.png
```

---

## 💡 常见问题速查

### Q1: IMDb 下载太慢？

```bash
export HF_ENDPOINT=https://hf-mirror.com
python prepare_imdb.py
```

或使用国内镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: MacBook Air 会不会太慢？

**不会！** 本项目专为 CPU 优化：
- ✅ 基线模型：5-8 分钟
- ✅ 领域自适应：6-7 分钟每个
- ✅ 全部实验：30-40 分钟
- ✅ 性能：75-82% 准确率

MacBook Air (M1/M2) 更快！

### Q3: TripAdvisor 数据列名不对？

编辑 `prepare_tripadvisor.py`，在 `text_col` 和 `rating_col` 查找部分添加你的列名。

### Q4: 训练时 MacBook 发热？

正常现象，建议：
- 放在通风处
- 关闭其他应用
- 训练时间不长（<1小时），不会损害电脑

### Q5: 想快速测试代码？

使用小数据集测试（用于调试）：

```python
# 在 prepare_imdb.py 的最后添加
train_df = train_df.sample(1000)
test_df = test_df.sample(500)
```

### Q6: 需要运行 BERT 吗？

**不需要！** 传统模型已经足够：
- ✅ 满足实验要求
- ✅ 性能优秀（75-82%）
- ✅ CPU 可运行
- ✅ 训练快速

BERT 是可选的增强实验（需要 GPU）。

---

## 📝 实验报告提纲

1. **引言**（1 段）
   - 领域自适应的重要性
   - 本实验的目标

2. **数据集**（1 页）
   - IMDb 和 TripAdvisor 的介绍
   - 数据预处理方法
   - 情感分布对比图

3. **方法**（2 页）
   - 基线模型：SVM/NB
   - 领域自适应：合并训练、BERT 两阶段微调
   - 超参数设置

4. **实验结果**（2 页）
   - 性能对比表格
   - 混淆矩阵（2-3 个）
   - 不同数据量的对比图

5. **错误分析**（1 页）
   - 采样 10-20 个错误样本
   - 分析错误类型（领域词汇、否定等）

6. **讨论**（1 页）
   - 为什么领域自适应有效
   - 数据效率分析
   - 实际应用价值

7. **结论**（1 段）
   - 总结主要发现
   - 局限性和未来工作

---

## 🎓 评分要点

根据实验要求，重点关注：

1. ✅ **数据预处理**：两个领域数据的清洗和标签映射
2. ✅ **基线模型**：至少一个基线（SVM 或 NB）
3. ✅ **领域自适应**：至少一种自适应方法
4. ✅ **性能对比**：准确率和 F1 分数
5. ✅ **混淆矩阵**：至少一个
6. ✅ **错误分析**：采样分析常见错误
7. ✅ **情感分布对比**：源域 vs 目标域
8. ✅ **讨论**：领域自适应的价值和挑战

---

## 💻 代码结构速查

```
src/
├── prepare_imdb.py         # IMDb 预处理
├── prepare_tripadvisor.py  # TripAdvisor 预处理
├── train_traditional.py    # 传统模型（SVM/NB）
│   └── --method baseline/combined
│   └── --model svm/nb
│   └── --target_ratio 1pct/5pct/10pct
├── train_bert.py           # BERT 两阶段微调
│   └── --stage 1/2/eval
│   └── --target_ratio 1pct/5pct/10pct
└── evaluate.py             # 统一评估和可视化
```

---

**祝实验顺利！有问题参考 README.md 或 DATA_GUIDE.md** 🎉

