#!/usr/bin/env python3
"""
Run Gaussian Blur Experiment on MIOTCD Dataset

Applies Gaussian blur to all compression levels and evaluates
with car vs background classifier.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image, ImageFilter
import pandas as pd
from tqdm import tqdm
import shutil

import sys
sys.path.append('../../model')
from models import get_resnet

# Configuration
COMPRESSION_LEVELS = [
    'filtered_images_sigma_0.5_quality_BAD',
    'filtered_images_sigma_1_quality_BAD',
    'filtered_images_sigma_1.5_quality_BAD',
    'filtered_images_sigma_2_quality_BAD',
    'filtered_images_sigma_2.5_quality_BAD',
    'filtered_images_sigma_3_quality_BAD',
    'filtered_images_sigma_3.5_quality_BAD',
    'filtered_images_sigma_4_quality_BAD',
    'filtered_images_sigma_4.5_quality_BAD',
    'filtered_images_sigma_5_quality_BAD'
]

BLUR_RADII = [0.5, 1.0, 2.0]  # Different blur strengths to test
MODEL_PATH = '../../model/resnet18_binary_best.pth'  # Using working car vs background model
BASE_DIR = Path('../data/filtered_images_quality_BAD')
OUTPUT_BASE = Path('../data/miotcd_blurred_2')
RESULTS_FILE = 'gaussian_blur_results2.csv'


def apply_gaussian_blur_to_dataset(input_dir, output_dir, radius):
    """Apply Gaussian blur to all images in dataset"""
    print(f"  Applying Gaussian blur (radius={radius}) to {input_dir.name}...")

    # Create output structure
    for subdir in ['car', 'background']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Process each class
    total_images = 0
    for class_name in ['car', 'background']:
        input_class_dir = input_dir / class_name
        output_class_dir = output_dir / class_name

        images = list(input_class_dir.glob('*'))
        for img_path in images:
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.ppm']:
                # Load, blur, save
                img = Image.open(img_path)
                blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
                blurred.save(output_class_dir / img_path.name)
                total_images += 1

    print(f"    ✓ Processed {total_images} images")
    return total_images


def evaluate_dataset(model, data_dir, device):
    """Evaluate model on a dataset"""
    # Transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    overall_acc = 100 * correct / total
    bg_acc = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    car_acc = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    return overall_acc, bg_acc, car_acc


def main():
    print("="*70)
    print("GAUSSIAN BLUR EXPERIMENT ON MIOTCD")
    print("="*70)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")

    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = get_resnet('resnet18', num_classes=2, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle different model architectures (backbone/head prefixes)
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove backbone. and head. prefixes if present
        new_key = k.replace('backbone.', '').replace('head.', '')

        # Map numbered layers to named layers for backbone architecture
        if new_key.startswith('0.'):
            new_key = new_key.replace('0.', 'conv1.')
        elif new_key.startswith('1.'):
            new_key = new_key.replace('1.', 'bn1.')
        elif new_key.startswith('4.'):
            new_key = new_key.replace('4.', 'layer1.')
        elif new_key.startswith('5.'):
            new_key = new_key.replace('5.', 'layer2.')
        elif new_key.startswith('6.'):
            new_key = new_key.replace('6.', 'layer3.')
        elif new_key.startswith('7.'):
            new_key = new_key.replace('7.', 'layer4.')

        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")

    # Results storage
    results = []

    # Process each compression level
    for comp_level in COMPRESSION_LEVELS:
        print(f"\n{'='*70}")
        print(f"Processing: {comp_level}")
        print(f"{'='*70}")

        input_dir = BASE_DIR / comp_level
        if not input_dir.exists():
            print(f"  ⚠ Directory not found, skipping")
            continue

        # Evaluate original (no blur)
        print(f"\n  Evaluating ORIGINAL (no blur)...")
        overall, bg, car = evaluate_dataset(model, input_dir, device)
        results.append({
            'compression_level': comp_level,
            'blur_radius': 0.0,
            'overall_accuracy': overall,
            'background_accuracy': bg,
            'car_accuracy': car
        })
        print(f"    Overall: {overall:.2f}% | Background: {bg:.2f}% | Car: {car:.2f}%")

        # Apply and evaluate each blur radius
        for radius in BLUR_RADII:
            output_dir = OUTPUT_BASE / f"{comp_level}_blur_{radius}"

            # Apply blur
            apply_gaussian_blur_to_dataset(input_dir, output_dir, radius)

            # Evaluate
            print(f"  Evaluating BLURRED (radius={radius})...")
            overall, bg, car = evaluate_dataset(model, output_dir, device)
            results.append({
                'compression_level': comp_level,
                'blur_radius': radius,
                'overall_accuracy': overall,
                'background_accuracy': bg,
                'car_accuracy': car
            })
            print(f"    Overall: {overall:.2f}% | Background: {bg:.2f}% | Car: {car:.2f}%")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Blurred datasets saved to: {OUTPUT_BASE}/")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(df.to_string(index=False))

    # Best results
    print("\n" + "="*70)
    print("BEST RESULTS")
    print("="*70)
    best_overall = df.loc[df['overall_accuracy'].idxmax()]
    print(f"\nBest Overall Accuracy: {best_overall['overall_accuracy']:.2f}%")
    print(f"  Compression: {best_overall['compression_level']}")
    print(f"  Blur radius: {best_overall['blur_radius']}")

    best_car = df.loc[df['car_accuracy'].idxmax()]
    print(f"\nBest Car Accuracy: {best_car['car_accuracy']:.2f}%")
    print(f"  Compression: {best_car['compression_level']}")
    print(f"  Blur radius: {best_car['blur_radius']}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
