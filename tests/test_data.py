"""
Unit tests for data loading and preprocessing
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import HAM10000Dataset, get_dataloaders, CLASS_NAMES
from src.data.preprocessing import SkinLesionPreprocessor


class TestHAM10000Dataset:
    """Test HAM10000Dataset class"""
    
    def test_class_names(self):
        """Test that class names are correctly defined"""
        assert len(CLASS_NAMES) == 7
        assert 'mel' in CLASS_NAMES
        assert 'nv' in CLASS_NAMES
    
    def test_dataset_init(self):
        """Test dataset initialization"""
        # This would require actual data
        # Mock test for structure
        assert HAM10000Dataset is not None


class TestDataLoader:
    """Test dataloader creation"""
    
    def test_dataloader_function_exists(self):
        """Test that get_dataloaders function exists"""
        assert callable(get_dataloaders)


class TestSkinLesionPreprocessor:
    """Test preprocessing utilities"""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization"""
        preprocessor = SkinLesionPreprocessor('D:/dl_data/data')
        assert preprocessor is not None
    
    def test_class_weights_shape(self):
        """Test class weights computation"""
        # Mock test - would need actual data
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
