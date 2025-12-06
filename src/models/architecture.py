"""
Model architecture definitions using EfficientNet and ResNet
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class SkinLesionClassifier(nn.Module):
    """
    Skin Lesion Classifier using EfficientNet or ResNet backbone
    
    Args:
        architecture: Base architecture ('efficientnet_b0', 'efficientnet_b3', 'resnet50', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        architecture: str = 'efficientnet_b3',
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        
        # Load backbone
        if 'efficientnet' in architecture:
            self.backbone = self._create_efficientnet(architecture, pretrained)
            in_features = self.backbone.classifier[1].in_features
            # Remove default classifier
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in architecture:
            self.backbone = self._create_resnet(architecture, pretrained)
            in_features = self.backbone.fc.in_features
            # Remove default fc layer
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_efficientnet(self, architecture: str, pretrained: bool):
        """Create EfficientNet backbone"""
        if architecture == 'efficientnet_b0':
            if pretrained:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            else:
                weights = None
            return models.efficientnet_b0(weights=weights)
        elif architecture == 'efficientnet_b1':
            if pretrained:
                weights = models.EfficientNet_B1_Weights.DEFAULT
            else:
                weights = None
            return models.efficientnet_b1(weights=weights)
        elif architecture == 'efficientnet_b2':
            if pretrained:
                weights = models.EfficientNet_B2_Weights.DEFAULT
            else:
                weights = None
            return models.efficientnet_b2(weights=weights)
        elif architecture == 'efficientnet_b3':
            if pretrained:
                weights = models.EfficientNet_B3_Weights.DEFAULT
            else:
                weights = None
            return models.efficientnet_b3(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {architecture}")
    
    def _create_resnet(self, architecture: str, pretrained: bool):
        """Create ResNet backbone"""
        if architecture == 'resnet50':
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            else:
                weights = None
            return models.resnet50(weights=weights)
        elif architecture == 'resnet101':
            if pretrained:
                weights = models.ResNet101_Weights.DEFAULT
            else:
                weights = None
            return models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet variant: {architecture}")
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_num_params(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    architecture: str = 'efficientnet_b3',
    num_classes: int = 7,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = 'cuda'
) -> SkinLesionClassifier:
    """
    Create and initialize model
    
    Args:
        architecture: Model architecture
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout: Dropout rate
        device: Device to load model on
        
    Returns:
        Initialized model
    """
    model = SkinLesionClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    model = model.to(device)
    
    print(f"\nModel: {architecture}")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    return model


def load_checkpoint(
    model: SkinLesionClassifier,
    checkpoint_path: str,
    device: str = 'cuda',
    strict: bool = True
) -> SkinLesionClassifier:
    """
    Load model from checkpoint with PyTorch 2.6+ compatibility
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Model with loaded weights
    """
    # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Val Accuracy: {checkpoint.get('val_accuracy', 0):.2f}%")
    else:
        model.load_state_dict(checkpoint, strict=strict)
        print(f"✓ Loaded model weights from {checkpoint_path}")
    
    return model


def save_checkpoint(
    model: SkinLesionClassifier,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    save_path: str,
    **kwargs
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Validation loss
        val_accuracy: Validation accuracy
        save_path: Path to save checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'architecture': model.architecture,
        'num_classes': model.num_classes,
        **kwargs
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")
