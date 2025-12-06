"""
Model performance monitoring
"""
import mlflow
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np


class ModelMonitor:
    """
    Monitor model performance in production
    """
    
    def __init__(self, model_name: str = "skin_lesion_classifier"):
        self.model_name = model_name
        self.metrics_history = []
        
    def log_prediction(
        self,
        prediction: str,
        confidence: float,
        true_label: str = None,
        metadata: Dict = None
    ):
        """
        Log a single prediction
        
        Args:
            prediction: Predicted class
            confidence: Prediction confidence
            true_label: Ground truth label (if available)
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': float(confidence),
            'true_label': true_label,
            'metadata': metadata or {}
        }
        
        self.metrics_history.append(log_entry)
        
        # Log to MLflow if available
        try:
            mlflow.log_metric('prediction_confidence', confidence)
            if true_label:
                is_correct = prediction == true_label
                mlflow.log_metric('prediction_correct', int(is_correct))
        except:
            pass
    
    def compute_performance_metrics(self) -> Dict:
        """
        Compute performance metrics from logged predictions
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.metrics_history:
            return {}
        
        # Filter predictions with ground truth
        labeled_predictions = [
            entry for entry in self.metrics_history 
            if entry['true_label'] is not None
        ]
        
        if not labeled_predictions:
            return {
                'total_predictions': len(self.metrics_history),
                'avg_confidence': np.mean([e['confidence'] for e in self.metrics_history])
            }
        
        # Calculate accuracy
        correct = sum(
            1 for entry in labeled_predictions 
            if entry['prediction'] == entry['true_label']
        )
        accuracy = correct / len(labeled_predictions)
        
        # Average confidence
        avg_confidence = np.mean([e['confidence'] for e in labeled_predictions])
        
        return {
            'total_predictions': len(self.metrics_history),
            'labeled_predictions': len(labeled_predictions),
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'correct_predictions': correct
        }
    
    def save_logs(self, filepath: str):
        """Save prediction logs to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"✓ Logs saved to {filepath}")
    
    def load_logs(self, filepath: str):
        """Load prediction logs from JSON file"""
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        print(f"✓ Loaded {len(self.metrics_history)} log entries")
