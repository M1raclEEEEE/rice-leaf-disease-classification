由于数据集包含4000多张图片（约400MB），无法直接上传到GitHub。
手动下载
数据集链接：https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset



自动下载：
```bash
# 1. 运行下载说明
python download_data.py

# 2. 下载zip文件（从Kaggle或百度网盘）

# 3. 解压（把zip文件放到项目文件夹后运行）
python download_data.py --extract rice-plant-diseases-dataset.zip

# 4. 检查
python check_data.py
