# src/inference.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

sys.path.append('.')

class RiceDiseaseClassifier:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.classes = ['Bacterialblight', 'Brownspot', 'Leafsmut']
        
        # 与训练时相同的transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        # 需要导入模型架构
        from main import improve_resnet_model
        model = improve_resnet_model(num_classes=3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image_path):
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.classes[predicted_class],
            'confidence': confidence,
            'probabilities': {cls: prob.item() for cls, prob in zip(self.classes, probabilities[0])}
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='水稻叶病分类推理')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='模型路径')
    
    args = parser.parse_args()
    
    classifier = RiceDiseaseClassifier(args.model)
    result = classifier.predict(args.image)
    
    print(f"预测结果: {result['class']}")
    print(f"置信度: {result['confidence']:.2%}")
    print("\n各类别概率:")
    for cls, prob in result['probabilities'].items():
        print(f"  {cls}: {prob:.2%}")
