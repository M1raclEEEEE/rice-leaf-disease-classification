#!/usr/bin/env python3
"""
检查数据集是否完整
运行: python check_data.py
"""

import os

print("🔍 检查数据集...")

# 检查文件夹
folders = [
    "rice leaf diseases dataset",
    "rice leaf diseases dataset/Bacterialblight",
    "rice leaf diseases dataset/Brownspot",
    "rice leaf diseases dataset/Leafsmut"
]

all_ok = True
for folder in folders:
    path = os.path.join("data", folder)
    if os.path.exists(path):
        print(f"✅ {folder}")
    else:
        print(f"❌ {folder}")
        all_ok = False

print("\n" + "="*40)
if all_ok:
    print("✅ 数据集文件夹结构正确！")
    print("可以运行主程序: python src/main.py")
else:
    print("❌ 数据集不完整")
    print("请运行: python data/download_data.py")
