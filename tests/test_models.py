"""
Unit tests for model architecture
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.architecture import SkinLesionClassifier, create_model


class TestSkinLesionClassifier:
    """Test model architecture"""
    
    def test_model_creation(self):
        """Test model can be created"""
        model = SkinLesionClassifier(
            architecture='efficientnet_b3',
            num_classes=7,
            pretrained=False,
            dropout=0.3
        )
        assert model is not None
        assert model.num_classes == 7
    
    def test_model_forward_pass(self):
        """Test forward pass with dummy input"""
        model = SkinLesionClassifier(
            architecture='efficientnet_b3',
            num_classes=7,
            pretrained=False
        )
        model.eval()
        
        # Create dummy input
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 7)
    
    def test_model_parameters(self):
        """Test model has correct number of parameters"""
        model = SkinLesionClassifier(
            architecture='efficientnet_b3',
            num_classes=7,
            pretrained=False
        )
        
        num_params = model.get_num_params()
        assert num_params > 0
        assert num_params > 1_000_000  # Should have millions of parameters
    
    def test_create_model_function(self):
        """Test create_model utility function"""
        model = create_model(
            architecture='efficientnet_b3',
            num_classes=7,
            pretrained=False,
            device='cpu'
        )
        assert model is not None
        assert next(model.parameters()).device.type == 'cpu'


class TestModelArchitectures:
    """Test different model architectures"""
    
    @pytest.mark.parametrize("arch", [
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b3',
    ])
    def test_efficientnet_variants(self, arch):
        """Test different EfficientNet variants"""
        model = SkinLesionClassifier(
            architecture=arch,
            num_classes=7,
            pretrained=False
        )
        assert model is not None
        assert model.architecture == arch


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
