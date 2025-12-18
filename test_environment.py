#!/usr/bin/env python
"""
环境测试脚本
运行此脚本检查所有依赖是否正确安装
"""

import sys

def test_imports():
    """测试所有必要的包是否能正确导入"""
    print("正在测试 Python 环境...\n")
    print(f"Python 版本: {sys.version}")
    print()
    
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('datasets', 'Datasets'),
        ('bs4', 'BeautifulSoup4'),
        ('tqdm', 'TQDM'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    optional_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
    ]
    
    success = True
    results = []
    
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', '未知版本')
            results.append((name, '✓', version))
            print(f"✓ {name:20s} {version}")
        except ImportError as e:
            results.append((name, '✗', str(e)))
            print(f"✗ {name:20s} 未安装")
            success = False
    
    print()
    
    # 检查可选包
    print("可选依赖（仅 BERT 实验需要）:")
    optional_available = True
    for package, name in optional_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', '未知版本')
            print(f"✓ {name:20s} {version}")
        except ImportError:
            print(f"- {name:20s} 未安装（可选）")
            optional_available = False
    
    print()
    
    if success:
        print("=" * 50)
        print("✓ 核心依赖包已正确安装！")
        if not optional_available:
            print("\n注意: PyTorch/Transformers 未安装")
            print("      CPU 实验不需要这些包，可以正常运行")
            print("      如需运行 BERT: pip install torch transformers")
        print("=" * 50)
    else:
        print("=" * 50)
        print("✗ 部分核心依赖包未安装，请运行:")
        print("  pip install -r requirements.txt")
        print("=" * 50)
    
    return success

def test_gpu():
    """测试 GPU 是否可用"""
    print("\n正在测试 GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU 可用: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"  说明: 可以运行 BERT 实验（可选）")
        else:
            print("ℹ️  GPU 不可用，将使用 CPU")
            print("  说明: 本项目专为 CPU 优化，传统模型运行快速")
            print("       不需要 GPU 即可完成全部实验要求")
    except ImportError:
        print("ℹ️  PyTorch 未安装（CPU 实验不需要）")
        print("  说明: 传统模型（SVM/NB）性能优秀，无需深度学习")

def test_data():
    """测试数据目录是否存在"""
    print("\n正在检查数据目录...")
    
    from pathlib import Path
    
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'results',
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path:30s} 存在")
        else:
            print(f"⚠ {dir_path:30s} 不存在（将自动创建）")

def test_scripts():
    """测试脚本文件是否存在"""
    print("\n正在检查脚本文件...")
    
    from pathlib import Path
    
    scripts = [
        'src/prepare_imdb.py',
        'src/prepare_tripadvisor.py',
        'src/train_traditional.py',
        'src/train_bert.py',
        'src/evaluate.py',
    ]
    
    all_exist = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} 缺失")
            all_exist = False
    
    if all_exist:
        print("\n✓ 所有脚本文件完整")
    else:
        print("\n✗ 部分脚本文件缺失")

def main():
    """主函数"""
    print("=" * 50)
    print("领域自适应情感分析 - 环境测试")
    print("=" * 50)
    print()
    
    # 测试依赖包
    imports_ok = test_imports()
    
    # 测试 GPU
    if imports_ok:
        test_gpu()
    
    # 测试数据目录
    test_data()
    
    # 测试脚本文件
    test_scripts()
    
    print("\n" + "=" * 50)
    print("环境测试完成！")
    print("=" * 50)
    print("\n下一步:")
    print("1. 如果依赖包有问题，运行: pip install -r requirements.txt")
    print("2. 阅读 QUICKSTART.md 开始实验")
    print("3. 准备数据: python src/prepare_imdb.py")
    print()

if __name__ == '__main__':
    main()

