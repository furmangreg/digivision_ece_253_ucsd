#!/usr/bin/env python3
"""
Evaluate Binary Car vs Pickup Truck Classifier

Usage:
    python evaluate_binary_classifier.py \
      --model-path model/resnet18_binary_best.pth \
      --data-dir data/car_vs_truck/test
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from models import get_resnet

class CustomHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)
        print(f"- CustomHead initialized with {in_features} input features and {num_classes} output classes")
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = CustomHead(512, num_classes)
        print(f"- CustomModel initialized with {num_classes} classes")
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def evaluate_binary_classifier(model, dataloader, device):
    """Evaluate binary classifier"""
    model.eval()

    correct = 0
    total = 0

    # Per-class metrics
    class_correct = [0, 0]
    class_total = [0, 0]

    # Confusion matrix
    confusion = [[0, 0], [0, 0]]  # [true][predicted]

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class and confusion matrix
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()

                class_total[true_label] += 1
                if pred_label == true_label:
                    class_correct[true_label] += 1

                confusion[true_label][pred_label] += 1

    # Calculate metrics
    accuracy = 100 * correct / total
    car_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    truck_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    return accuracy, car_acc, truck_acc, confusion, class_total


def main():
    parser = argparse.ArgumentParser(description='Evaluate binary classifier')
    parser.add_argument('--model-path', required=True,
                        help='Path to trained binary model')
    parser.add_argument('--model-name', default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])
    #parser.add_argument('--data-dir', default='data/car_vs_truck/test',
    #                    help='Test data directory')
    parser.add_argument('--data-dir', default='collected_images/collected_images',
                        help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='auto')

    args = parser.parse_args()

    # Device setup
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using device: CUDA")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using device: MPS")
        else:
            device = torch.device('cpu')
            print(f"Using device: CPU")
    else:
        device = torch.device(args.device)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    backbone = models.resnet18(weights='IMAGENET1K_V1')
    backbone = nn.Sequential(*list(backbone.children())[:-2])

    model = CustomModel(backbone, 2)
    model.avgpool = nn.AdaptiveAvgPool2d(1)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("✓ Model loaded")

    # Load test data
    print(f"\nLoading test data from {args.data_dir}...")
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        root=args.data_dir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"  Test images: {len(test_dataset)}")
    print(f"  Classes: {test_dataset.classes}")

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    accuracy, car_acc, truck_acc, confusion, class_total = evaluate_binary_classifier(
        model, test_loader, device
    )

    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"\nPer-Class Accuracy:")
    print(f"  Background (class 0):         {car_acc:.2f}%")
    print(f"  Car        (class 1):         {truck_acc:.2f}%")

    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Background    Car")
    print(f"True Background {confusion[0][0]:4d}          {confusion[0][1]:4d}   ({class_total[0]} total)")
    print(f"     Car        {confusion[1][0]:4d}          {confusion[1][1]:4d}   ({class_total[1]} total)")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
