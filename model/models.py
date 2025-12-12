"""
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path


def get_resnet(model_name='resnet18', num_classes=100, pretrained=True):
    """
    Get a ResNet model configured for TinyImageNet
    Args:
        model_name: 'resnet18'
        num_classes: Number of output classes 
        pretrained: Use ImageNet pretrained weights
    """

    # Map model names to torchvision models
    model_dict = {
        'resnet18': models.resnet18,
    }

    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(model_dict.keys())}")

    # Load model
    if pretrained:
        print(f"Loading {model_name} with ImageNet pretrained weights...")
        model = model_dict[model_name](pretrained=True)
    else:
        print(f"Creating {model_name} from scratch...")
        model = model_dict[model_name](pretrained=False)

    # Modify final layer for CIFAR-100 (100 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f"Modified final layer: {num_features} -> {num_classes} classes")

    return model


def load_model(checkpoint_path, model_name='resnet18', num_classes=100, device='cpu'):
    """
    Load a trained model from checkpoint
    Args:
        checkpoint_path: Path to .pth checkpoint file
        model_name: Architecture name
        num_classes: Number of classes
        device: Device to load on ('cpu' or 'cuda')
    """
    # Create model architecture
    model = get_resnet(model_name, num_classes, pretrained=False)

    # Load weights
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Loaded model from epoch {checkpoint['epoch']}")
            if 'accuracy' in checkpoint:
                print(f"Model accuracy: {checkpoint['accuracy']:.2f}%")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model


def save_model(model, save_path, epoch=None, accuracy=None, optimizer=None):
    """
    Args:
        model: PyTorch model
        save_path: Path to save checkpoint
        epoch: Current epoch (optional)
        accuracy: Current accuracy (optional)
        optimizer: Optimizer state (optional)
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def get_model_summary(model, input_size=(1, 3, 64, 64)):
    """
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
    """
    from torchsummary import summary
    try:
        summary(model, input_size[1:])
    except:
        print("Install torchsummary for detailed model summary: pip install torchsummary")
        print(f"\nModel: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# Convenience functions
def create_resnet18(pretrained=True):
    """Quick function to create ResNet18 for CIFAR-100"""
    return get_resnet('resnet18', num_classes=100, pretrained=pretrained)
