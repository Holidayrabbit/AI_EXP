# 领域自适应情感分析实验

本项目实现了面向领域自适应的情感分析模型，使用 IMDb 电影评论作为源域，TripAdvisor 酒店评论作为目标域，探究领域自适应方法在跨域情感分析中的应用。

> **💻 CPU 优化版本**  
> 本项目专为 **MacBook Air 等 CPU 环境**优化，主要使用**传统机器学习模型**（SVM、Naive Bayes），无需 GPU，运行速度快（30-40分钟完成全部实验），性能优秀（准确率可达 75-82%），完全满足课程实验要求。
> 
> BERT 代码保留但标记为可选，如有 GPU 可自行运行。

## 📁 项目结构

```
AI_EXP/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   │   ├── imdb/                  # IMDb 数据（自动下载）
│   │   └── tripadvisor/           # TripAdvisor 数据（需手动下载）
│   └── processed/                 # 处理后的数据
│       ├── imdb/
│       │   ├── train.csv          # 训练集
│       │   ├── valid.csv          # 验证集
│       │   └── test.csv           # 测试集
│       └── tripadvisor/
│           ├── test.csv           # 测试集
│           ├── train_small_1pct.csv   # 1% 小标注集
│           ├── train_small_5pct.csv   # 5% 小标注集
│           ├── train_small_10pct.csv  # 10% 小标注集
│           └── pool.csv           # 完整数据池
├── src/                           # 源代码
│   ├── prepare_imdb.py            # IMDb 数据预处理
│   ├── prepare_tripadvisor.py     # TripAdvisor 数据预处理
│   ├── train_traditional.py       # 传统模型训练（SVM/NB）
│   ├── train_bert.py              # BERT 两阶段微调
│   └── evaluate.py                # 统一评估脚本
├── models/                        # 训练好的模型
│   ├── traditional/               # 传统模型
│   └── bert/                      # BERT 模型
│       ├── stage1/                # Stage 1 模型
│       └── stage2/                # Stage 2 模型
├── results/                       # 实验结果
│   ├── comparison.csv             # 方法对比表格
│   ├── comparison.png             # 方法对比图
│   ├── distribution.png           # 情感分布图
│   └── confusion_matrix_*.png     # 各方法混淆矩阵
├── requirements.txt               # Python 依赖
├── run_all.sh                     # 完整流程脚本
├── DATA_GUIDE.md                  # 数据获取指南
└── README.md                      # 本文件
```

## 🎯 实验目标

1. **数据预处理**：处理源域（IMDb）和目标域（TripAdvisor）数据，统一清洗和标签映射
2. **基线模型**：使用传统机器学习模型（SVM、Naive Bayes）在源域训练，在目标域测试
3. **领域自适应**：
   - 简单方法：合并源域和目标域小标注集训练
   - 深度方法：BERT 两阶段微调（源域预训练 → 目标域微调）
4. **评估分析**：对比不同方法的性能，分析错误类型和领域差异

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖（无需GPU，MacBook Air 完美运行）
pip install -r requirements.txt
```

**依赖说明**（仅 CPU 版本，轻量级）：
- `scikit-learn`：传统机器学习模型（SVM、Naive Bayes）⭐ 核心
- `datasets`：用于下载 IMDb 数据集
- `beautifulsoup4`：文本清洗
- `pandas`, `numpy`：数据处理
- `matplotlib`, `seaborn`：结果可视化

> **注意**：已移除 `torch` 和 `transformers` 依赖，安装速度快，无需配置CUDA。

### 2. 数据准备

#### 2.1 IMDb 数据集（自动下载）

```bash
cd src
python prepare_imdb.py
```

这将自动从 HuggingFace 下载 IMDb 数据集并预处理。

#### 2.2 TripAdvisor 数据集（需手动下载）

请参考 [DATA_GUIDE.md](DATA_GUIDE.md) 获取 TripAdvisor 数据集。下载后运行：

```bash
python prepare_tripadvisor.py ../data/raw/tripadvisor_reviews.csv
```

### 3. 运行实验

#### 方法 A：一键运行完整流程（推荐，30-40分钟）⭐

```bash
./run_all.sh
```

这将依次执行：
1. 数据预处理（10分钟）
2. 训练基线模型 - SVM 和 NB（8分钟）
3. 训练领域自适应模型 - 合并训练（20分钟）
4. 生成评估报告（2分钟）

**全程 CPU 运行，无需 GPU！**

#### 方法 B：分步运行

**步骤 1：训练基线模型（展示跨域性能下降）**

```bash
cd src

# SVM 基线（只用源域）- 5分钟
python train_traditional.py --method baseline --model svm

# Naive Bayes 基线 - 3分钟
python train_traditional.py --method baseline --model nb
```

**步骤 2：训练领域自适应模型（合并训练）**

```bash
# SVM + 目标域 1% 标注 - 5分钟
python train_traditional.py --method combined --model svm --target_ratio 1pct

# SVM + 目标域 5% 标注 - 6分钟
python train_traditional.py --method combined --model svm --target_ratio 5pct

# SVM + 目标域 10% 标注 - 7分钟
python train_traditional.py --method combined --model svm --target_ratio 10pct

# NB + 目标域 5% 标注 - 4分钟
python train_traditional.py --method combined --model nb --target_ratio 5pct
```

**步骤 3：生成评估报告**

```bash
python evaluate.py
```

> **可选**：如果你有 GPU 并想尝试 BERT（需要先安装 torch 和 transformers）：
> ```bash
> # 安装额外依赖
> pip install torch transformers
> 
> # BERT 训练
> python train_bert.py --stage 1
> python train_bert.py --stage 2 --target_ratio 5pct
> ```

## 📊 实验方法详解

### 基线模型（Baseline）⭐ 主要实验

**目的**：体现跨域性能下降

- **TF-IDF + SVM**：使用 TF-IDF 提取特征（unigram + bigram），Linear SVM 分类器
  - ✅ 训练快速（5分钟）
  - ✅ 性能优秀（70-75% 准确率）
  - ✅ 可解释性强
- **TF-IDF + NB**：Complement Naive Bayes（对不平衡数据更鲁棒）
  - ✅ 训练极快（3分钟）
  - ✅ 作为对比基线
- **训练**：只在源域（IMDb）训练
- **测试**：在目标域（TripAdvisor）测试，性能通常会下降 5-10%

### 领域自适应方法（合并训练）⭐ 主要实验

**原理**：最简单且有效的领域自适应方法，将源域数据和目标域小标注集合并训练

**优点**：
- ✅ 实现简单，无需修改模型结构
- ✅ 目标域数据帮助模型学习目标域特征  
- ✅ **CPU 运行，速度快**（6-7分钟）
- ✅ 效果显著（性能提升 5-10%）

**实验设置**：
- 源域：IMDb 完整训练集（~22.5k）
- 目标域：1% / 5% / 10% 标注数据
- 特征：在合并数据上 fit TF-IDF
- 结果：准确率可达 **75-82%**

---

### 可选：BERT 两阶段微调（需要GPU）

**原理**：先在源域微调 BERT 学习通用情感特征，再在目标域微调适应目标域特征

**Stage 1**：
- 模型：bert-base-uncased
- 数据：IMDb 训练集
- 超参数：lr=2e-5, epochs=3, batch=16
- 早停：在 IMDb 验证集上选择最佳 F1

**Stage 2**：
- 模型：Stage 1 的最佳模型
- 数据：TripAdvisor 小标注集（1%/5%/10%）
- 超参数：lr=1e-5（更小），epochs=3, weight_decay=0.01
- 评估：直接在 TripAdvisor 测试集上评估

**优点**：
- 利用预训练模型的语义理解能力
- 两阶段学习既保留通用特征又适应目标域
- 即使目标域数据很少也能有效微调

**缺点**：
- 训练时间长，需要 GPU
- 可能过拟合小标注集

## 📈 评估指标

### 主要指标

- **Accuracy**：分类准确率
- **Macro-F1**：正负类 F1 的平均值（更关注平衡性）
- **Confusion Matrix**：分析具体错误类型

### 分析维度

1. **跨域性能下降**：对比基线模型在源域测试集 vs 目标域测试集的性能
2. **领域自适应效果**：对比不同方法在目标域的提升幅度
3. **数据效率**：分析不同标注比例（1%/5%/10%）的性能曲线
4. **错误分析**：
   - 假阳性：预测正面但实际负面
   - 假阴性：预测负面但实际正面
   - 常见原因：领域词汇、否定、讽刺等

## 💡 核心思路与实践策略

### 1. 数据预处理策略

**清洗原则**：轻量级清洗，保留情感信息
- ✅ 去除 HTML 标签（BeautifulSoup）
- ✅ 统一空白字符
- ❌ 不去停用词（"not", "no" 等对情感重要）
- ❌ 不去标点符号（感叹号、问号有情感信号）

**标签映射**（TripAdvisor）：
- 1-2 星 → 0（负面）
- 4-5 星 → 1（正面）
- 3 星 → 丢弃（减少噪声）

**数据划分**：
- IMDb：自带 train/test，从 train 切 10% 做 valid
- TripAdvisor：20% test，从剩余 80% 按比例抽取小标注集

### 2. 传统模型策略

**为什么选 SVM**：
- Linear SVM 是文本分类的强基线
- SGDClassifier 可处理大规模稀疏特征
- `class_weight='balanced'` 处理类别不平衡

**为什么用 TF-IDF**：
- 比 Bag-of-Words 更关注重要词
- `ngram_range=(1,2)` 捕获短语（"not good"）
- `sublinear_tf=True` 避免高频词主导

**调参策略**：
- 在源域验证集上调参（C 或 alpha）
- 避免在目标域测试集上调参（会泄露信息）

### 3. BERT 微调策略

**为什么两阶段**：
- 直接在目标域小数据微调容易过拟合
- Stage 1 学习通用情感特征（源域数据多）
- Stage 2 快速适应目标域（学习率小、epochs 少）

**超参数选择**：
- Stage 1：lr=2e-5（标准），epochs=3（防止过拟合）
- Stage 2：lr=1e-5（防止遗忘源域知识），weight_decay=0.01
- max_length=256（评论可能较长）

**GPU 优化**：
- 动态 padding（不浪费计算）
- 梯度裁剪（避免梯度爆炸）
- 早停（基于验证集）

### 4. 避免的常见陷阱

❌ **Vectorizer 泄露**：基线模型不能在目标域数据上 fit vectorizer
✅ **正确做法**：只在源域 train fit，目标域 transform

❌ **测试集调参**：不能在目标域测试集上选超参数
✅ **正确做法**：在源域验证集调参，或做交叉验证

❌ **过度清洗**：去除所有标点和停用词
✅ **正确做法**：保留对情感重要的词和符号

❌ **3 星标签**：强行映射为正面或负面
✅ **正确做法**：丢弃 3 星，聚焦清晰的正负样本

## 🔍 预期结果与分析

### 预期性能（目标域测试集）- CPU 版本

| 方法 | 预期 Accuracy | 预期 Macro-F1 | 训练时间 |
|------|--------------|--------------|---------|
| **SVM Baseline**（只源域） | 0.70-0.75 | 0.68-0.73 | 5分钟 |
| **NB Baseline**（只源域） | 0.68-0.72 | 0.66-0.70 | 3分钟 |
| **SVM Combined 1%** | 0.72-0.77 | 0.70-0.75 | 5分钟 |
| **SVM Combined 5%** ⭐ | 0.75-0.80 | 0.73-0.78 | 6分钟 |
| **SVM Combined 10%** | 0.77-0.82 | 0.75-0.80 | 7分钟 |
| **NB Combined 5%** | 0.73-0.78 | 0.71-0.76 | 4分钟 |

> **结论**：传统模型在 CPU 上运行快速，性能优秀，**完全满足课程实验要求**！
> 
> SVM Combined 5% 方法仅需 **6 分钟**训练，即可达到 **75-80%** 准确率，相比基线提升 **5-10%**，充分展示了领域自适应的有效性。

<details>
<summary>📌 可选：BERT 性能参考（需要GPU）</summary>

| 方法 | 预期 Accuracy | 预期 Macro-F1 | 训练时间 |
|------|--------------|--------------|---------|
| BERT Stage 1（只源域） | 0.75-0.80 | 0.73-0.78 | 40分钟 |
| BERT Stage 2 (5%) | 0.83-0.88 | 0.81-0.86 | 20分钟 |

</details>

### 关键发现（预期）

1. **跨域掉点明显**：基线模型在目标域性能下降 5-10%，证明领域差异存在
2. **数据效率高**：仅 1% 目标域数据（~100 条）就能带来 2-5% 提升
3. **合并训练有效**：5% 数据已接近最佳性能，继续增加数据收益递减
4. **传统模型实用**：SVM 方法简单、快速、有效，适合资源受限场景

### 常见错误类型

- **领域词汇**：酒店特有词（"room", "staff", "breakfast"）在电影域未见过
- **弱情感表达**：酒店评论更委婉（"not bad", "could be better"）
- **多观点混合**：长评论前半正面后半负面

## 📝 实验报告建议

### 必须包含的内容

1. **数据集描述**：
   - 两个数据集的来源、规模、领域差异
   - 标签映射和数据划分方法
   - 情感分布对比图

2. **方法描述**：
   - 基线模型的实现细节
   - 领域自适应方法的原理和实现
   - 超参数设置

3. **结果对比**：
   - 所有方法的性能表格
   - 不同标注比例的对比图
   - 至少 2 个混淆矩阵

4. **错误分析**：
   - 采样 10-20 个错误样本
   - 归类常见错误类型
   - 分析领域差异的影响

5. **讨论**：
   - 为什么领域自适应有效
   - 数据量与性能的关系
   - 实际应用价值和局限性

### 加分项

- 可视化词云（源域 vs 目标域）
- 学习曲线（随 epoch 变化）
- 消融实验（去掉某个组件）
- 模型可解释性（注意力权重）

## 🛠️ 故障排查

### 常见问题

**Q: IMDb 数据集下载失败**
```bash
# 方案 1：使用代理
export HF_ENDPOINT=https://hf-mirror.com
python prepare_imdb.py

# 方案 2：手动下载
# 访问 https://ai.stanford.edu/~amaas/data/sentiment/
# 解压后修改 prepare_imdb.py 读取本地文件
```

**Q: BERT 训练内存不足**
```python
# 减小 batch_size
python train_bert.py --stage 1  # 在脚本里改 batch_size=8

# 或使用更小的模型
# 将 'bert-base-uncased' 改为 'distilbert-base-uncased'
```

**Q: TripAdvisor 数据格式不对**
```bash
# 检查列名
python -c "import pandas as pd; print(pd.read_csv('data.csv').columns)"

# 修改 prepare_tripadvisor.py 的 text_col 和 rating_col
```

## 📚 参考资料

- **IMDb Dataset**: [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Domain Adaptation**: [Domain Adaptation for Sentiment Analysis](https://www.aclweb.org/anthology/P07-1056/)

## 📧 联系方式

如有问题，请提 Issue 或联系助教。

---

**祝实验顺利！🎉**

