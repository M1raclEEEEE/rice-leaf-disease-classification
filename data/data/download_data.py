#!/usr/bin/env python3
"""
水稻病害数据集一键下载脚本
运行: python download_data.py
"""

print("="*60)
print("🌾 水稻病害数据集一键下载工具")
print("="*60)

print("""
请按以下步骤操作：

📥 第1步：下载数据集文件
------------------------
1. 点击链接下载：
   https://pan.baidu.com/s/1EXAMPLE123456  (提取码:1234)

   或

   从Kaggle下载：
   https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset

2. 下载文件：rice-plant-diseases-dataset.zip
   （约400MB）

📁 第2步：运行自动解压
------------------------
把下载的zip文件放到本项目文件夹，然后运行：
python data/download_data.py --file rice-plant-diseases-dataset.zip

   或直接运行：
python data/download_data.py
然后按照提示操作

✅ 第3步：检查是否成功
------------------------
运行：python data/check_data.py
看到"✅ 数据集完整"表示成功！
""")

print("="*60)
print("💡 如果有问题，请联系: your_email@example.com")
print("="*60)

# 简单的自动解压功能
import sys
import zipfile
import os

if len(sys.argv) > 1 and sys.argv[1] == "--file":
    zip_path = sys.argv[2] if len(sys.argv) > 2 else "rice-plant-diseases-dataset.zip"
    
    if os.path.exists(zip_path):
        print(f"\n正在解压: {zip_path}")
        os.makedirs("data", exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("data")
            print("✅ 解压完成！")
            print("数据集已放在: data/rice leaf diseases dataset/")
        except:
            print("❌ 解压失败，请手动解压")
    else:
        print(f"❌ 找不到文件: {zip_path}")
