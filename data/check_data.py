import os

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

if all_ok:
    print("\n✅ 数据集完整！")
else:
    print("\n❌ 数据集不完整，请运行: python data/download_data.py")
