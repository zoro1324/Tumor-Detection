"""
Swin Transformer Model for Brain Tumor Classification

Implements Swin Transformer architecture with transfer learning from pretrained models.
Uses timm library for easy access to pretrained weights.
"""

import torch
import torch.nn as nn
from timm import create_model
from typing import Dict, Any, Optional


class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer for brain tumor classification.
    
    Supports various Swin variants from timm:
    - swin_tiny_patch4_window7_224
    - swin_small_patch4_window7_224
    - swin_base_patch4_window7_224
    - swin_large_patch4_window7_224
    """
    
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize Swin Transformer classifier.
        
        Args:
            model_name: Name of the Swin model from timm
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            dropout: Dropout rate for classification head
        """
        super(SwinTransformerClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Create model using timm
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all parameters except the classification head."""
        for name, param in self.model.named_parameters():
            if 'head' not in name:  # Don't freeze the classification head
                param.requires_grad = False
        
        print(f"Frozen backbone parameters. Only training classification head.")
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        print("Unfrozen all parameters for full fine-tuning.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        return self.model(x)
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: Optional[str] = None):
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            x: Input tensor (B, C, H, W)
            layer_name: Specific layer to extract (None = last layer)
            
        Returns:
            Feature maps from specified layer
        """
        features = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output
            return hook
        
        # Register hooks
        hooks = []
        if layer_name:
            # Register hook on specific layer
            for name, module in self.model.named_modules():
                if layer_name in name:
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
        else:
            # Register hook on last layer before head
            for name, module in self.model.named_modules():
                if 'norm' in name and 'head' not in name:
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return features
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


def create_swin_model(config: Dict[str, Any]) -> SwinTransformerClassifier:
    """
    Factory function to create Swin Transformer model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SwinTransformerClassifier instance
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    model = SwinTransformerClassifier(
        model_name=model_config.get('architecture', 'swin_tiny_patch4_window7_224'),
        num_classes=data_config.get('num_classes', 4),
        pretrained=model_config.get('pretrained', True),
        freeze_backbone=model_config.get('freeze_backbone', False),
        dropout=model_config.get('dropout', 0.1)
    )
    
    return model


# Available Swin Transformer models
SWIN_MODELS = {
    'swin_tiny': 'swin_tiny_patch4_window7_224',
    'swin_small': 'swin_small_patch4_window7_224',
    'swin_base': 'swin_base_patch4_window7_224',
    'swin_large': 'swin_large_patch4_window7_224',
}


if __name__ == "__main__":
    # Test Swin Transformer model
    print("Testing Swin Transformer Model...")
    
    # Create model
    model = SwinTransformerClassifier(
        model_name='swin_tiny_patch4_window7_224',
        num_classes=4,
        pretrained=False,  # Set to False for testing (faster)
        dropout=0.1
    )
    
    # Print model info
    params = model.count_parameters()
    print(f"\nModel: swin_tiny_patch4_window7_224")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output[0].detach().numpy()}")
    
    # Test freezing/unfreezing
    print("\nTesting freeze/unfreeze...")
    model._freeze_backbone()
    frozen_params = model.count_parameters()
    print(f"After freezing - Trainable: {frozen_params['trainable']:,}")
    
    model.unfreeze_backbone()
    unfrozen_params = model.count_parameters()
    print(f"After unfreezing - Trainable: {unfrozen_params['trainable']:,}")
    
    print("\n✓ Swin Transformer model test passed!")
