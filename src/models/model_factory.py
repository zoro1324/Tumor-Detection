"""
Model Factory

Unified interface for creating different model architectures.
"""

import torch.nn as nn
from typing import Dict, Any

from .vit_model import ViTClassifier, create_vit_model, VIT_MODELS
from .swin_transformer import SwinTransformerClassifier, create_swin_model, SWIN_MODELS


class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> nn.Module:
        """
        Create a model based on configuration.
        
        Args:
            config: Configuration dictionary with 'model' and 'data' sections
            
        Returns:
            PyTorch model
        """
        model_config = config.get('model', {})
        architecture = model_config.get('architecture', 'vit_base_patch16_224')
        
        # Determine which model type
        if 'vit' in architecture.lower():
            return create_vit_model(config)
        elif 'swin' in architecture.lower():
            return create_swin_model(config)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    @staticmethod
    def list_available_models() -> Dict[str, str]:
        """
        List all available model architectures.
        
        Returns:
            Dictionary of available models
        """
        return {
            **VIT_MODELS,
            **SWIN_MODELS
        }


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Convenience function to create model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch model
    """
    return ModelFactory.create_model(config)


if __name__ == "__main__":
    # Test model factory
    print("Testing Model Factory...")
    
    # List available models
    models = ModelFactory.list_available_models()
    print("\nAvailable models:")
    for key, value in models.items():
        print(f"  {key}: {value}")
    
    # Test creating ViT model
    print("\n=== Testing ViT Creation ===")
    vit_config = {
        'model': {
            'architecture': 'vit_base_patch16_224',
            'pretrained': False,
            'dropout': 0.1
        },
        'data': {
            'num_classes': 4
        }
    }
    
    vit_model = ModelFactory.create_model(vit_config)
    print(f"Created: {vit_model.__class__.__name__}")
    print(f"Parameters: {vit_model.count_parameters()['total']:,}")
    
    # Test creating Swin model
    print("\n=== Testing Swin Creation ===")
    swin_config = {
        'model': {
            'architecture': 'swin_tiny_patch4_window7_224',
            'pretrained': False,
            'dropout': 0.1
        },
        'data': {
            'num_classes': 4
        }
    }
    
    swin_model = ModelFactory.create_model(swin_config)
    print(f"Created: {swin_model.__class__.__name__}")
    print(f"Parameters: {swin_model.count_parameters()['total']:,}")
    
    print("\n✓ Model factory test passed!")
