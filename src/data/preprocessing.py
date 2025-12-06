"""
Data preprocessing utilities for skin lesion images
"""
import os
from pathlib import Path
from typing import Tuple, Optional
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm


class SkinLesionPreprocessor:
    """
    Preprocessing utilities for HAM10000 dataset
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def extract_dataset(self, zip_path: str, extract_to: Optional[str] = None):
        """
        Extract HAM10000 dataset from zip file
        
        Args:
            zip_path: Path to zip file
            extract_to: Target directory (default: same as data_dir)
        """
        if extract_to is None:
            extract_to = self.data_dir
        
        extract_to = Path(extract_to)
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete!")
        
    def analyze_dataset(self, metadata_file: str) -> pd.DataFrame:
        """
        Analyze dataset statistics and class distribution
        
        Args:
            metadata_file: Path to metadata CSV
            
        Returns:
            DataFrame with analysis results
        """
        df = pd.read_csv(metadata_file)
        
        print("\n=== Dataset Analysis ===")
        print(f"Total samples: {len(df)}")
        print(f"\nClass distribution:")
        print(df['dx'].value_counts())
        
        print(f"\nAge statistics:")
        print(df['age'].describe())
        
        print(f"\nSex distribution:")
        print(df['sex'].value_counts())
        
        print(f"\nLocalization distribution:")
        print(df['localization'].value_counts())
        
        # Check for duplicates
        duplicates = df['lesion_id'].value_counts()
        duplicates = duplicates[duplicates > 1]
        print(f"\nDuplicate lesions: {len(duplicates)}")
        
        return df
    
    def check_image_quality(
        self,
        image_dir: str,
        metadata_file: str,
        output_file: str = 'image_quality_report.csv'
    ):
        """
        Check image quality metrics: size, aspect ratio, corruption
        
        Args:
            image_dir: Directory containing images
            metadata_file: Path to metadata CSV
            output_file: Output CSV file for quality report
        """
        image_dir = Path(image_dir)
        df = pd.read_csv(metadata_file)
        
        quality_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
            image_id = row['image_id']
            image_path = image_dir / f"{image_id}.jpg"
            
            if not image_path.exists():
                quality_data.append({
                    'image_id': image_id,
                    'exists': False,
                    'width': None,
                    'height': None,
                    'aspect_ratio': None,
                    'corrupted': True
                })
                continue
            
            try:
                img = Image.open(image_path)
                width, height = img.size
                aspect_ratio = width / height
                
                quality_data.append({
                    'image_id': image_id,
                    'exists': True,
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'corrupted': False
                })
            except Exception as e:
                quality_data.append({
                    'image_id': image_id,
                    'exists': True,
                    'width': None,
                    'height': None,
                    'aspect_ratio': None,
                    'corrupted': True
                })
        
        quality_df = pd.DataFrame(quality_data)
        quality_df.to_csv(output_file, index=False)
        
        print(f"\n=== Image Quality Report ===")
        print(f"Missing images: {(~quality_df['exists']).sum()}")
        print(f"Corrupted images: {quality_df['corrupted'].sum()}")
        print(f"\nImage dimensions:")
        print(quality_df[['width', 'height']].describe())
        
        return quality_df
    
    def remove_hair(self, image: np.ndarray) -> np.ndarray:
        """
        Remove hair artifacts from dermoscopic images using morphological operations
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Processed image with hair removed
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        
        # Black hat transform to detect hair
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Threshold to create mask
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpaint to remove hair
        result = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)
        
        return result
    
    def compute_class_weights(self, metadata_file: str) -> dict:
        """
        Compute class weights for imbalanced dataset
        
        Args:
            metadata_file: Path to metadata CSV
            
        Returns:
            Dictionary mapping class indices to weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        from .dataset import CLASS_NAMES
        
        df = pd.read_csv(metadata_file)
        df['label'] = df['dx'].map(CLASS_NAMES)
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df['label']),
            y=df['label']
        )
        
        weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print("\n=== Class Weights ===")
        for i, weight in weight_dict.items():
            print(f"Class {i}: {weight:.4f}")
        
        return weight_dict
    
    def create_train_val_split(
        self,
        metadata_file: str,
        output_dir: str,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42
    ):
        """
        Create and save train/val/test splits
        
        Args:
            metadata_file: Path to metadata CSV
            output_dir: Directory to save split files
            val_split: Validation proportion
            test_split: Test proportion
            seed: Random seed
        """
        from sklearn.model_selection import train_test_split
        from .dataset import CLASS_NAMES
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_csv(metadata_file)
        df['label'] = df['dx'].map(CLASS_NAMES)
        
        # Stratified split
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_split + test_split),
            stratify=df['label'],
            random_state=seed
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_split / (val_split + test_split),
            stratify=temp_df['label'],
            random_state=seed
        )
        
        # Save splits
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        print(f"\n=== Splits Created ===")
        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
        print(f"\nFiles saved to {output_dir}")
