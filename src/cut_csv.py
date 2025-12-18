"""
CSV文件随机采样脚本
从大型CSV文件中随机保留指定行数的数据
"""
import pandas as pd
import random
from pathlib import Path

def cut_csv(input_file, output_file, sample_size=40000, seed=42):
    """
    从CSV文件中随机采样指定数量的行
    
    参数:
    - input_file: 输入CSV文件路径
    - output_file: 输出CSV文件路径
    - sample_size: 保留的行数
    - seed: 随机种子
    """
    random.seed(seed)
    
    print(f"读取文件: {input_file}")
    df = pd.read_csv(input_file)
    print(f"原始数据行数: {len(df)}")
    
    if len(df) <= sample_size:
        print(f"数据行数({len(df)})小于等于采样数量({sample_size})，保留全部数据")
        sampled_df = df
    else:
        print(f"随机采样 {sample_size} 行数据...")
        sampled_df = df.sample(n=sample_size, random_state=seed)
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存采样后的数据
    sampled_df.to_csv(output_file, index=False)
    print(f"采样完成，保存到: {output_file}")
    print(f"最终数据行数: {len(sampled_df)}")

if __name__ == '__main__':
    import sys
    
    # 默认参数
    input_file = '../data/raw/tripadvisor_reviews.csv'
    output_file = '../data/raw/tripadvisor_reviews_40k.csv'
    sample_size = 40000
    
    # 支持命令行参数
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    if len(sys.argv) >= 4:
        sample_size = int(sys.argv[3])
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"采样数量: {sample_size}")
    
    try:
        cut_csv(input_file, output_file, sample_size)
    except FileNotFoundError:
        print(f"错误: 文件不存在 {input_file}")
    except Exception as e:
        print(f"错误: {e}")
