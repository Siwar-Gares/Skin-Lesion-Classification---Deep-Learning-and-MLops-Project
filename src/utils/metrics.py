"""
Metrics calculation utilities
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> Dict:
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Per-class metrics
    if class_names:
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            metrics[f'precision_{class_name}'] = float(precision_per_class[i])
            metrics[f'recall_{class_name}'] = float(recall_per_class[i])
            metrics[f'f1_{class_name}'] = float(f1_per_class[i])
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = None,
    normalize: bool = False
) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize the matrix
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    return fig


def save_metrics_json(metrics: Dict, filepath: str):
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {filepath}")


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    """Print detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(report)
