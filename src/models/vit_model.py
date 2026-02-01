"""
Vision Transformer (ViT) Model for Brain Tumor Classification

Implements ViT architecture with transfer learning from pretrained models.
Uses timm library for easy access to pretrained weights.
"""

import torch
import torch.nn as nn
from timm import create_model
from typing import Dict, Any


class ViTClassifier(nn.Module):
    """
    Vision Transformer for brain tumor classification.
    
    Supports various ViT variants from timm:
    - vit_tiny_patch16_224
    - vit_small_patch16_224
    - vit_base_patch16_224
    - vit_large_patch16_224
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize ViT classifier.
        
        Args:
            model_name: Name of the ViT model from timm
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            dropout: Dropout rate for classification head
        """
        super(ViTClassifier, self).__init__()
        
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
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1):
        """
        Extract attention weights for visualization.
        
        Args:
            x: Input tensor (B, C, H, W)
            layer_idx: Which transformer block to extract from (-1 = last)
            
        Returns:
            Attention weights
        """
        # This requires hooking into the model's attention mechanism
        # Implementation depends on timm version and model structure
        # For simplicity, we'll implement a basic version
        
        # Enable hooks for attention extraction
        attentions = []
        
        def hook_fn(module, input, output):
            if hasattr(output, 'attn'):
                attentions.append(output.attn)
        
        # Register hook on transformer blocks
        hooks = []
        for block in self.model.blocks:
            hook = block.attn.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if len(attentions) > 0:
            return attentions[layer_idx]
        return None
    
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


def create_vit_model(config: Dict[str, Any]) -> ViTClassifier:
    """
    Factory function to create ViT model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ViTClassifier instance
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    model = ViTClassifier(
        model_name=model_config.get('architecture', 'vit_base_patch16_224'),
        num_classes=data_config.get('num_classes', 4),
        pretrained=model_config.get('pretrained', True),
        freeze_backbone=model_config.get('freeze_backbone', False),
        dropout=model_config.get('dropout', 0.1)
    )
    
    return model


# Available ViT models
VIT_MODELS = {
    'vit_tiny': 'vit_tiny_patch16_224',
    'vit_small': 'vit_small_patch16_224',
    'vit_base': 'vit_base_patch16_224',
    'vit_large': 'vit_large_patch16_224',
}


if __name__ == "__main__":
    # Test ViT model
    print("Testing Vision Transformer Model...")
    
    # Create model
    model = ViTClassifier(
        model_name='vit_base_patch16_224',
        num_classes=4,
        pretrained=False,  # Set to False for testing (faster)
        dropout=0.1
    )
    
    # Print model info
    params = model.count_parameters()
    print(f"\nModel: vit_base_patch16_224")
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
    
    print("\n✓ ViT model test passed!")
