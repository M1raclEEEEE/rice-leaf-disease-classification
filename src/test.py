import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def improve_resnet_model(num_classes=3):
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
    )
    return model

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model_path = "../models/best_model.pth"
    if not Path(model_path).exists():
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行: python scripts/download_model.py")
        return
    
    print(f"加载模型: {model_path}")
    model = improve_resnet_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    root = Path("/root/.cache/kagglehub/datasets/jay7080dev/rice-plant-diseases-dataset/versions/1")
    dataset_root = root / "rice leaf diseases dataset"
    
    if not dataset_root.exists():
        print(f"错误: 找不到数据集 {dataset_root}")
        print("请先运行: python data/download_data.py")
        return
    
    image_data_test_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = datasets.ImageFolder(
        root=dataset_root,
        transform=image_data_test_transform
    )
    
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    test_idx = indices[train_size + val_size:]
    test_dataset = Subset(full_dataset, test_idx)
    
    test_dataloader = DataLoader(test_dataset, batch_size=16,
                                shuffle=False, num_workers=0)
    
    print(f"测试集大小: {len(test_dataset)} 张图片")
    print(f"类别: {full_dataset.classes}")
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    accuracy = (all_preds == all_labels).mean()
    
    print(f"\n总体准确率: {accuracy*100:.2f}%")
    print(f"正确/总数: {np.sum(all_preds == all_labels)}/{len(all_labels)}")
    
    print("\n每个类别准确率:")
    class_names = full_dataset.classes
    for i, cls in enumerate(class_names):
        cls_idx = np.where(all_labels == i)[0]
        if len(cls_idx) > 0:
            cls_acc = (all_preds[cls_idx] == i).mean()
            print(f"{cls}: {cls_acc*100:.2f}% ({len(cls_idx)}张)")
    
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    
    if accuracy >= 0.9986:
        print("✅ 达到论文报告的99.86%准确率！")
    else:
        print(f"⚠️  当前准确率 ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    test()
