"""
Model training utilities
"""
import os
import time
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.architecture import save_checkpoint


class ModelTrainer:
    """
    Trainer class for model training and evaluation
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: str = 'cuda',
        experiment_name: str = 'skin_lesion_classification'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': running_loss / (batch_idx + 1)
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds) * 100
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float, Dict]:
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds) * 100
        
        # Compute detailed metrics
        metrics = {
            'accuracy': val_acc,
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_weighted': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        
        return val_loss, val_acc, metrics
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = 'checkpoints',
        early_stopping_patience: int = 10,
        save_best_only: bool = True
    ):
        """
        Train model for specified number of epochs
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            save_best_only: Only save best model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                'architecture': self.model.architecture,
                'num_epochs': num_epochs,
                'batch_size': self.train_loader.batch_size,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'optimizer': self.optimizer.__class__.__name__,
                'device': self.device
            })
            
            patience_counter = 0
            
            for epoch in range(1, num_epochs + 1):
                print(f"\nEpoch {epoch}/{num_epochs}")
                print("-" * 60)
                
                epoch_start = time.time()
                
                # Train
                train_loss, train_acc = self.train_epoch()
                
                # Validate
                val_loss, val_acc, val_metrics = self.validate()
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = self.optimizer.param_groups[0]['lr']
                
                epoch_time = time.time() - epoch_start
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['lr'].append(current_lr)
                
                # Log to MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'learning_rate': current_lr,
                    'precision_macro': val_metrics['precision_macro'],
                    'recall_macro': val_metrics['recall_macro'],
                    'f1_macro': val_metrics['f1_macro']
                }, step=epoch)
                
                # Print epoch results
                print(f"\nResults:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
                print(f"  F1-Score (Macro): {val_metrics['f1_macro']:.4f}")
                print(f"  Learning Rate: {current_lr:.6f}")
                print(f"  Time: {epoch_time:.2f}s")
                
                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        val_loss=val_loss,
                        val_accuracy=val_acc,
                        save_path=checkpoint_path,
                        metrics=val_metrics
                    )
                    
                    print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")
                else:
                    patience_counter += 1
                
                # Save last checkpoint
                if not save_best_only:
                    checkpoint_path = os.path.join(save_dir, 'last_model.pth')
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        val_loss=val_loss,
                        val_accuracy=val_acc,
                        save_path=checkpoint_path
                    )
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠ Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                    break
            
            # Plot and save training curves
            self._plot_training_curves(save_dir)
            
            # Log best model
            mlflow.log_artifact(os.path.join(save_dir, 'best_model.pth'))
            mlflow.log_artifact(os.path.join(save_dir, 'training_curves.png'))
            
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Best Val Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
            print(f"{'='*60}\n")
    
    def _plot_training_curves(self, save_dir: str):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curve
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curve
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[2].plot(epochs, self.history['lr'], 'g-')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
        plt.close()
