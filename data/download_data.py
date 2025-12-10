import os
import zipfile
import sys

def main():
    print("="*60)
    print("水稻病害数据集下载工具")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        zip_path = sys.argv[2] if len(sys.argv) > 2 else "rice-plant-diseases-dataset.zip"
        
        if os.path.exists(zip_path):
            print(f"解压: {zip_path}")
            os.makedirs("data", exist_ok=True)
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall("data")
                print("✅ 解压完成！数据集在: data/rice leaf diseases dataset/")
            except:
                print("❌ 解压失败，请手动解压")
        else:
            print(f"❌ 找不到文件: {zip_path}")
        return
    
    print("""
下载步骤：
1. 从Kaggle下载: https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset
2. 下载文件: rice-plant-diseases-dataset.zip
3. 放到项目文件夹
4. 运行: python data/download_data.py --extract rice-plant-diseases-dataset.zip

if __name__ == "__main__":
    main()
