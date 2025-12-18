# 数据获取指南

本文档指导如何获取实验所需的两个数据集：IMDb 和 TripAdvisor。

---

## 1. IMDb 数据集（源域）- 自动下载 ✅

### 方法 A：使用脚本自动下载（推荐）

```bash
cd src
python prepare_imdb.py
```

脚本会自动从 HuggingFace 下载 IMDb Large Movie Review Dataset 并预处理。

### 方法 B：如果自动下载失败

**原因**：网络问题、HuggingFace 访问受限

**解决方案 1：使用镜像**

```bash
export HF_ENDPOINT=https://hf-mirror.com
python prepare_imdb.py
```

**解决方案 2：手动下载**

1. 访问官方网站：https://ai.stanford.edu/~amaas/data/sentiment/
2. 下载 `aclImdb_v1.tar.gz` (84.1 MB)
3. 解压到 `data/raw/imdb/` 目录
4. 修改 `prepare_imdb.py` 读取本地文件（见下方代码）

```python
# 在 prepare_imdb.py 中修改
def prepare_imdb_local(data_dir='../data'):
    """从本地文件读取 IMDb"""
    import tarfile
    
    raw_dir = Path(data_dir) / 'raw' / 'imdb'
    tar_file = raw_dir / 'aclImdb_v1.tar.gz'
    
    if not tar_file.exists():
        print(f"错误：文件不存在 {tar_file}")
        return
    
    # 解压
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(raw_dir)
    
    # 读取训练集
    train_dir = raw_dir / 'aclImdb' / 'train'
    train_data = []
    
    for label, label_name in [(0, 'neg'), (1, 'pos')]:
        label_dir = train_dir / label_name
        for file in label_dir.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                text = clean_text(f.read())
                train_data.append({'text': text, 'label': label})
    
    # ... 类似处理测试集 ...
```

### 数据集信息

- **来源**：ACL 2011, Andrew Maas et al.
- **规模**：50,000 条电影评论
  - 训练集：25,000 条（正负各半）
  - 测试集：25,000 条（正负各半）
- **标签**：二分类（positive / negative）
- **语言**：英文
- **平均长度**：~230 词

---

## 2. TripAdvisor 数据集（目标域）- 需手动获取 ⚠️

### 为什么需要手动获取

TripAdvisor 的评论数据受版权保护，不能直接提供。你需要：
1. 从公开数据集网站下载，或
2. 使用学术机构提供的数据，或
3. 自己爬取（需遵守 robots.txt 和服务条款）

### 推荐来源

#### 选项 1：Kaggle 公开数据集（推荐）⭐

**数据集名称**：TripAdvisor Hotel Reviews

**步骤**：

1. 访问 Kaggle：https://www.kaggle.com/
2. 搜索 "TripAdvisor Hotel Reviews" 或 "hotel reviews"
3. 推荐数据集：
   - `jiashenliu/515k-hotel-reviews-data-in-europe`（51万条，包含评分）
   - `datafiniti/hotel-reviews`（10万+条）
   - `andrewmvd/trip-advisor-hotel-reviews`（2万+条）

4. 下载 CSV 文件（需要 Kaggle 账号）

5. 确保 CSV 包含以下列：
   - 评论文本：`Review`, `Review_Text`, `review`, `text` 等
   - 评分：`Rating`, `rating`, `score` (1-5 星)

6. 将文件放到 `data/raw/` 目录，例如：
   ```
   data/raw/tripadvisor_reviews.csv
   ```

7. 运行预处理脚本：
   ```bash
   cd src
   python prepare_tripadvisor.py ../data/raw/tripadvisor_reviews.csv
   ```

#### 选项 2：UCI Machine Learning Repository

**步骤**：

1. 访问：https://archive.ics.uci.edu/
2. 搜索 "hotel" 或 "opinion"
3. 下载数据集
4. 按上述方法预处理

#### 选项 3：学术数据集

某些论文会公开数据集链接，搜索：
- "TripAdvisor sentiment dataset"
- "hotel review dataset sentiment analysis"
- 在 Google Scholar 或 Papers with Code 搜索

### 数据集要求

为了保证实验有效，数据集应满足：

1. **规模**：至少 5,000 条评论（推荐 10,000+）
2. **字段**：
   - ✅ 评论文本（英文）
   - ✅ 评分（1-5 星）或标签（正/负）
3. **质量**：
   - 评论长度合理（至少 10 词）
   - 不要全是 5 星好评（需要正负样本）
4. **授权**：允许学术/教学使用

### 如果列名不匹配怎么办

`prepare_tripadvisor.py` 会自动识别常见列名：
- 文本列：`Review`, `Review_Text`, `review`, `text`, `review_text`
- 评分列：`Rating`, `rating`, `score`

如果还是不匹配，手动修改：

```python
# 在 prepare_tripadvisor.py 中修改
text_col = 'your_text_column_name'  # 改成你的列名
rating_col = 'your_rating_column_name'  # 改成你的列名
```

---

## 3. 自己爬取数据（高级选项）🕷️

### ⚠️ 注意事项

1. **遵守 robots.txt**：检查网站是否允许爬取
2. **服务条款**：确保不违反网站的使用协议
3. **爬取频率**：设置合理延迟，避免给服务器造成压力
4. **个人学习**：仅用于学术研究，不得商用

### 方法 A：使用现成爬虫工具

```bash
# 安装 scrapy
pip install scrapy

# 使用公开的 TripAdvisor 爬虫（GitHub 搜索）
# 例如：https://github.com/search?q=tripadvisor+scraper
```

### 方法 B：编写简单爬虫

**注意**：以下仅为示例，TripAdvisor 实际页面结构可能不同，且可能有反爬措施。

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def scrape_tripadvisor_hotel(hotel_url, max_pages=10):
    """
    爬取单个酒店的评论
    警告：此代码仅为示例，实际使用需要处理反爬、翻页等
    """
    reviews = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for page in range(max_pages):
        try:
            # 添加延迟，避免被封
            time.sleep(random.uniform(2, 5))
            
            response = requests.get(hotel_url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 以下选择器仅为示例，需根据实际页面调整
            review_elements = soup.find_all('div', class_='review-container')
            
            for review in review_elements:
                # 提取评分（通常在 class 或 data 属性中）
                rating_elem = review.find('span', class_='ui_bubble_rating')
                rating = extract_rating(rating_elem)  # 需自己实现
                
                # 提取文本
                text_elem = review.find('p', class_='partial_entry')
                text = text_elem.get_text(strip=True) if text_elem else ''
                
                if text and rating:
                    reviews.append({
                        'Review': text,
                        'Rating': rating
                    })
            
            # 翻页逻辑（需根据实际情况）
            next_button = soup.find('a', class_='next')
            if not next_button:
                break
            hotel_url = 'https://www.tripadvisor.com' + next_button['href']
            
        except Exception as e:
            print(f"爬取出错: {e}")
            break
    
    return pd.DataFrame(reviews)

# 使用
# df = scrape_tripadvisor_hotel('酒店页面URL')
# df.to_csv('tripadvisor_reviews.csv', index=False)
```

### 方法 C：使用第三方 API（收费）

- **ScraperAPI**：https://www.scraperapi.com/
- **Apify**：https://apify.com/
- 这些服务提供现成的 TripAdvisor 爬虫

---

## 4. 数据验证

获取数据后，验证是否符合要求：

```bash
cd src
python -c "
import pandas as pd

# 读取数据
df = pd.read_csv('../data/raw/tripadvisor_reviews.csv')

print(f'数据集大小: {len(df)} 条')
print(f'列名: {df.columns.tolist()}')
print(f'评分分布:')
print(df['Rating'].value_counts().sort_index())
print(f'前 3 条样本:')
print(df.head(3))
"
```

**预期输出**：

```
数据集大小: 10000 条
列名: ['Review', 'Rating', ...]
评分分布:
1    1200
2    1500
3    2000
4    2800
5    2500
前 3 条样本:
...
```

**检查清单**：
- ✅ 数据量 >= 5000
- ✅ 有评论文本和评分
- ✅ 评分范围 1-5
- ✅ 每个评分都有样本（不要只有 5 星）
- ✅ 评论是英文（如果是多语言，需要过滤）

---

## 5. 故障排查

### 问题 1：数据集太小

**解决方案**：
- 下载多个数据集并合并
- 使用更大的公开数据集
- 适当降低实验要求（至少 3000 条）

### 问题 2：没有评分，只有标签

**解决方案**：
如果数据集已经有 positive/negative 标签：

```python
# 修改 prepare_tripadvisor.py
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
# 跳过评分映射步骤
```

### 问题 3：评论不是英文

**解决方案**：
- 优先使用英文数据集（因为 IMDb 是英文）
- 或使用翻译 API（Google Translate, DeepL）
- 或同时使用中文 IMDb 数据（需要多语言 BERT）

### 问题 4：爬虫被封

**解决方案**：
- 降低爬取频率
- 使用代理 IP
- 使用现成的公开数据集（推荐）

---

## 6. 快速测试

如果暂时没有 TripAdvisor 数据，可以先用 IMDb 的另一部分做测试：

```python
# 在 prepare_tripadvisor.py 中
def prepare_tripadvisor_dummy():
    """使用 IMDb 的一部分模拟 TripAdvisor（仅测试用）"""
    from datasets import load_dataset
    
    dataset = load_dataset('imdb')
    
    # 使用 unsupervised 数据的一部分
    # 注意：这不是真正的跨域实验，仅用于测试代码
    test_data = []
    for i, item in enumerate(dataset['test'][:5000]):
        if i % 2 == 0:  # 采样一半
            test_data.append({
                'text': item['text'],
                'label': item['label']
            })
    
    df = pd.DataFrame(test_data)
    # ... 后续处理 ...
```

**警告**：这种方法不能用于正式实验，因为没有真正的领域差异！

---

## 总结

1. **IMDb**：自动下载，无需手动操作 ✅
2. **TripAdvisor**：推荐从 Kaggle 下载公开数据集 ⭐
3. **验证**：确保数据质量和格式
4. **预处理**：运行 `prepare_tripadvisor.py`

**推荐流程**：
```bash
# 1. 准备 IMDb（自动）
python prepare_imdb.py

# 2. 从 Kaggle 下载 TripAdvisor 数据
# 3. 预处理 TripAdvisor
python prepare_tripadvisor.py ../data/raw/tripadvisor_reviews.csv

# 4. 验证数据
ls ../data/processed/imdb/
ls ../data/processed/tripadvisor/
```

如有问题，请检查文件格式和列名，或参考 README.md 的故障排查部分。

