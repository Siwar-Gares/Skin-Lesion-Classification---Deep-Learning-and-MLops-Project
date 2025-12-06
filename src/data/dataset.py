"""
HAM10000 Dataset Module
Handles data loading, preprocessing, and augmentation for skin lesion classification
"""
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# HAM10000 class mapping
CLASS_NAMES = {
    'akiec': 0,  # Actinic keratoses
    'bcc': 1,    # Basal cell carcinoma
    'bkl': 2,    # Benign keratosis-like lesions
    'df': 3,     # Dermatofibroma
    'mel': 4,    # Melanoma
    'nv': 5,     # Melanocytic nevi
    'vasc': 6    # Vascular lesions
}

CLASS_DESCRIPTIONS = {
    0: 'Actinic Keratoses (akiec)',
    1: 'Basal Cell Carcinoma (bcc)',
    2: 'Benign Keratosis (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic Nevi (nv)',
    6: 'Vascular Lesions (vasc)'
}


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesion images
    
    Args:
        data_dir: Path to dataset directory
        metadata_file: Path to CSV metadata file
        image_ids: List of image IDs to include
        transform: Albumentations transform pipeline
        return_metadata: Whether to return additional metadata
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        image_ids: Optional[List[str]] = None,
        transform: Optional[A.Compose] = None,
        return_metadata: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # Filter by image IDs if provided
        if image_ids is not None:
            self.metadata = self.metadata[self.metadata['image_id'].isin(image_ids)]
        
        self.metadata = self.metadata.reset_index(drop=True)
        
        # Map class names to indices
        self.metadata['label'] = self.metadata['dx'].map(CLASS_NAMES)
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        
        # Load image
        image_id = row['image_id']
        
        # Check both image directories (images are split into part_1 and part_2)
        possible_paths = [
            self.data_dir / "HAM10000_images_part_1" / f"{image_id}.jpg",
            self.data_dir / "HAM10000_images_part_2" / f"{image_id}.jpg",
        ]
        
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            # Try alternative extensions
            for base_path in possible_paths:
                for ext in ['.png', '.jpeg', '.JPG']:
                    alt_path = base_path.parent / f"{image_id}{ext}"
                    if alt_path.exists():
                        image_path = alt_path
                        break
                if image_path:
                    break
        
        if image_path is None:
            raise FileNotFoundError(
                f"Image not found: {image_id}\\n"
                f"Searched in:\\n" + "\\n".join([f"  - {p}" for p in possible_paths])
            )
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = torch.tensor(row['label'], dtype=torch.long)
        
        output = {
            'image': image,
            'label': label,
            'image_id': image_id
        }
        
        if self.return_metadata:
            output.update({
                'age': row.get('age', -1),
                'sex': row.get('sex', 'unknown'),
                'localization': row.get('localization', 'unknown'),
                'dx_type': row.get('dx_type', 'unknown')
            })
        
        return output


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Get augmentation pipeline for training
    Includes modern augmentation techniques: RandAugment-style transforms
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.2),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2
            ),
            A.CLAHE(),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Get transforms for validation/test sets"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_dataloaders(
    data_dir: str,
    metadata_file: str,
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders with stratified splits
    
    Args:
        data_dir: Path to image directory
        metadata_file: Path to CSV metadata
        batch_size: Batch size for dataloaders
        image_size: Target image size
        val_split: Validation set proportion
        test_split: Test set proportion
        num_workers: Number of dataloader workers
        seed: Random seed for reproducibility
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    metadata['label'] = metadata['dx'].map(CLASS_NAMES)
    
    # Stratified split
    train_ids, temp_ids = train_test_split(
        metadata['image_id'].values,
        test_size=(val_split + test_split),
        stratify=metadata['label'].values,
        random_state=seed
    )
    
    # Split temp into validation and test
    temp_metadata = metadata[metadata['image_id'].isin(temp_ids)]
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=test_split / (val_split + test_split),
        stratify=temp_metadata['label'].values,
        random_state=seed
    )
    
    # Create datasets
    train_dataset = HAM10000Dataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        image_ids=train_ids.tolist(),
        transform=get_train_transforms(image_size)
    )
    
    val_dataset = HAM10000Dataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        image_ids=val_ids.tolist(),
        transform=get_val_transforms(image_size)
    )
    
    test_dataset = HAM10000Dataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        image_ids=test_ids.tolist(),
        transform=get_val_transforms(image_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader
