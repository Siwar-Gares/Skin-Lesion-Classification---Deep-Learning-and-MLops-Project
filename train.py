"""
Training script for skin lesion classification
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow

from src.data.dataset import get_dataloaders, CLASS_NAMES
from src.data.preprocessing import SkinLesionPreprocessor
from src.models.architecture import create_model
from src.models.training import ModelTrainer
from src.utils.config import Config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train skin lesion classification model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='D:/dl_data/data',
                        help='Path to dataset directory')
    parser.add_argument('--metadata-file', type=str, default='D:/dl_data/data/HAM10000/HAM10000_metadata.csv',
                        help='Path to metadata CSV file')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='efficientnet_b3',
                        choices=['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                                'efficientnet_b3', 'resnet50', 'resnet101'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of dataloader workers')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--experiment-name', type=str, default='skin_lesion_classification',
                        help='MLflow experiment name')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main training function"""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        metadata_file=args.metadata_file,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Compute class weights for imbalanced dataset
    print("\nComputing class weights...")
    preprocessor = SkinLesionPreprocessor(args.data_dir)
    class_weights = preprocessor.compute_class_weights(args.metadata_file)
    
    # Convert to tensor and ensure correct order
    class_weights_list = [class_weights[i] for i in range(len(class_weights))]
    weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
    
    # Create model
    print(f"\nCreating model: {args.architecture}")
    model = create_model(
        architecture=args.architecture,
        num_classes=7,
        pretrained=args.pretrained,
        dropout=args.dropout,
        device=device
    )
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_name=args.experiment_name
    )
    
    # Train model
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train(
        num_epochs=args.epochs,
        save_dir=args.checkpoint_dir,
        early_stopping_patience=10,
        save_best_only=True
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60 + "\n")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_acc, test_metrics = trainer.validate()
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro): {test_metrics['recall_macro']:.4f}")
    
    print("\n✅ Training pipeline completed successfully!")
    print(f"📁 Best model saved to: {args.checkpoint_dir}/best_model.pth")
    print(f"🔬 MLflow tracking: mlflow ui --port 5000\n")


if __name__ == '__main__':
    main()
