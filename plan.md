# 实验方案设计

---

## 1) 数据集与标签定义（IMDb 源域，TripAdvisor 目标域）

### 1.1 源域：IMDb Large Movie Review Dataset（ACL 2011）
- 内容：25k train + 25k test（正/负各半），无标注的 unsup 另外 50k（可用于无监督域适配）
- 标签：原生二分类（pos/neg）
- 文本：英文长评

### 1.2 目标域：TripAdvisor Hotel Reviews（公开数据集）
选择标准：必须包含 `review text` + `rating(1-5)` 或 `label`，且允许研究/教学使用，避免直接爬官网。常见公开版本字段类似：
- `Review` / `Review_Text`
- `Rating`（1–5）

**二分类标签映射（强烈建议统一为二分类，便于跨域对齐）：**
- `rating ∈ {1,2} -> label=0 (neg)`
- `rating ∈ {4,5} -> label=1 (pos)`
- `rating = 3`：两种做法择一并写清楚
  A) 丢弃（推荐，减少噪声）
  B) 作为中性做三分类（实验量更大，不建议首次做）

### 1.3 数据划分（目标域要留 test，并做“小标注”）
- IMDb：用自带 train/test；从 train 再切 10% 做 valid
- TripAdvisor：
  - `target_test`：20%（只用于最终评估）
  - `target_pool`：剩余 80%
  - 从 `target_pool` 里抽少量标注做 `target_train_small`：1% / 5% / 10% 三档（用于微调/合并训练）
  - 其余 `target_unlabeled`：可用于无监督域适配（如 DANN 的 domain loss）

**抽样规则**：按 label 分层抽样（stratified），保证每档正负比例接近。

---

## 2) 统一预处理（传统模型与 BERT 两套管线）

### 2.1 轻清洗（两域一致）
- 去 HTML：BeautifulSoup 或正则
- 统一空白
- 不强行去停用词（对情感有帮助，如 “not”）
- 保留感叹号/问号

### 2.2 传统模型：TF-IDF（强基线）
- `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, sublinear_tf=True)`
- 重要：
  - **Baseline（只源训练）**：vectorizer 只在 IMDb_train fit
  - **合并训练/有目标标注**：vectorizer 在 (IMDb_train + target_train_small) fit（这是允许的，因为相当于你确实“拿到了目标标注”）

### 2.3 BERT：tokenizer
- `max_length=256`（酒店/电影长评都可能较长；显存不够用 128）
- 动态 padding
- 输出取 `[CLS]` 做分类

---

## 3) 必做基线：IMDb 训练 → TripAdvisor 测试（体现跨域掉点）

### 3.1 Baseline-1：MultinomialNB（或 ComplementNB）
- 输入：TF-IDF 或 Count
- 调参：`alpha ∈ {0.5, 1.0, 2.0}`（在 IMDb_valid 上选）

### 3.2 Baseline-2：Linear SVM（推荐主基线）
- `LinearSVC(C ∈ {0.1, 1, 10})` 或 `SGDClassifier(loss='hinge', class_weight='balanced')`
- 在 IMDb_valid 上调参

### 3.3 报告
- IMDb_test：Acc/Macro-F1
- TripAdvisor_target_test：Acc/Macro-F1 + Confusion Matrix
- 采样 20 条 target_test 错误例子做人工归类（领域词、否定、讽刺、设施词等）

---

## 4) 领域自适应

### **BERT 两阶段微调**
1) Stage-1：IMDb_train 微调 BERT（监督）
   - lr=2e-5, epochs=3, batch=16, warmup=0.1
   - 早停：IMDb_valid Macro-F1
2) Stage-2：在 `target_train_small` 继续微调
   - lr=1e-5, epochs=3（小数据避免过拟合）
   - 可加 weight decay=0.01、dropout 默认
3) 最终：评估 TripAdvisor_target_test

**对照实验（必须做表格）：**
- BERT(only IMDb) → target_test
- BERT(IMDb→target 1%) → target_test
- BERT(IMDb→target 5%) → target_test
- BERT(IMDb→target 10%) → target_test

---

## 5) 评估与分析（按实验要求逐项输出）

### 5.1 指标
- 重点：TripAdvisor_target_test 的 Accuracy、Macro-F1
- 同时给 IMDb_test 作为参考

### 5.2 混淆矩阵与常见错误类型（目标域为主）
建议你重点看：
- 设施/位置词（room, location, breakfast, staff）在电影域没见过导致的误判
- “not bad / no complaints / could be better” 等弱情感表达
- 长评论多观点：前半好评后半差评

### 5.3 情感分布对比（源 vs 目标）
- IMDb 原本严格平衡；TripAdvisor 可能偏正（真实评论常“正多负少”）
- 给出正负占比柱状图；说明分布差异对模型阈值/偏置的影响

---

## 6) 工程落地与复现规范（建议按这个目录写代码）
- `data/`：原始与处理后数据
- `src/`
  - `prepare_imdb.py`, `prepare_tripadvisor.py`（清洗、映射标签、划分）
  - `train_svm.py`（TF-IDF + SVM/NB）
  - `finetune_bert.py`（stage1/stage2）
  - `eval.py`（统一输出 Acc/F1/混淆矩阵/错误样本）
- 固定随机种子、记录依赖版本、保存 best checkpoint

---