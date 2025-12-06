"""
Configuration management
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """Configuration class for managing project settings"""
    
    # Default paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = Path("D:/dl_data/data")
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    METRICS_DIR = BASE_DIR / "metrics"
    PLOTS_DIR = BASE_DIR / "plots"
    
    # Data settings
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 0
    
    # Model settings
    ARCHITECTURE = "efficientnet_b3"
    NUM_CLASSES = 7
    DROPOUT = 0.3
    
    # Training settings
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10
    
    # Class names
    CLASS_NAMES = [
        'akiec',  # Actinic Keratoses and Intraepithelial Carcinoma
        'bcc',    # Basal Cell Carcinoma
        'bkl',    # Benign Keratosis
        'df',     # Dermatofibroma
        'mel',    # Melanoma
        'nv',     # Melanocytic Nevi
        'vasc'    # Vascular skin lesions
    ]
    
    CLASS_DESCRIPTIONS = {
        'akiec': 'Actinic Keratoses and Intraepithelial Carcinoma',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis-like Lesions',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Skin Lesions'
    }
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get full path to data file"""
        return cls.DATA_DIR / filename
    
    @classmethod
    def get_checkpoint_path(cls, filename: str) -> Path:
        """Get full path to checkpoint file"""
        return cls.CHECKPOINT_DIR / filename
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'data_dir': str(cls.DATA_DIR),
            'image_size': cls.IMAGE_SIZE,
            'batch_size': cls.BATCH_SIZE,
            'architecture': cls.ARCHITECTURE,
            'num_classes': cls.NUM_CLASSES,
            'dropout': cls.DROPOUT,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY
        }
    
    @classmethod
    def save_yaml(cls, filepath: str):
        """Save config to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(cls.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load_yaml(cls, filepath: str) -> Dict[str, Any]:
        """Load config from YAML file"""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
