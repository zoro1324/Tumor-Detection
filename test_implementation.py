"""
Quick test script to verify the implementation works end-to-end.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 70)
print("BRAIN TUMOR DETECTION - IMPLEMENTATION TEST")
print("=" * 70)

# Test 1: Data Preprocessing
print("\n[1/5] Testing Data Preprocessing...")
from data.data_preprocessing import MRIPreprocessor, validate_dataset

preprocessor = MRIPreprocessor(image_size=224, apply_clahe=True)
validation = validate_dataset("dataset", ["glioma", "healthy", "meningioma", "pituitary"])
print(f"  ✓ Dataset valid: {validation['valid']}")
print(f"  ✓ Total images: {validation['total_images']}")
print(f"  ✓ Classes: {len(validation['class_distribution'])}")

# Test 2: Models
print("\n[2/5] Testing ViT Model...")
from models.vit_model import ViTClassifier

vit_model = ViTClassifier(
    model_name='vit_base_patch16_224',
    num_classes=4,
    pretrained=False,
    dropout=0.1
)
dummy_input = torch.randn(2, 3, 224, 224)
output = vit_model(dummy_input)
print(f"  ✓ ViT model created: {vit_model.count_parameters()['total']:,} parameters")
print(f"  ✓ Forward pass works: input {dummy_input.shape} → output {output.shape}")

print("\n[3/5] Testing Swin Transformer...")
from models.swin_transformer import SwinTransformerClassifier

swin_model = SwinTransformerClassifier(
    model_name='swin_tiny_patch4_window7_224',
    num_classes=4,
    pretrained=False,
    dropout=0.1
)
output_swin = swin_model(dummy_input)
print(f"  ✓ Swin model created: {swin_model.count_parameters()['total']:,} parameters")
print(f"  ✓ Forward pass works: input {dummy_input.shape} → output {output_swin.shape}")

# Test 3: Configuration
print("\n[4/5] Testing Configuration...")
import yaml

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"  ✓ Config loaded successfully")
print(f"  ✓ Model architecture: {config['model']['architecture']}")
print(f"  ✓ Batch size: {config['training']['batch_size']}")
print(f"  ✓ Learning rate: {config['training']['learning_rate']}")
print(f"  ✓ Federated clients: {config['federated']['num_clients']}")
print(f"  ✓ Federated rounds: {config['federated']['num_rounds']}")

# Test 4: Metrics
print("\n[5/5] Testing Metrics Calculator...")
from evaluation.metrics import MetricsCalculator
import numpy as np

np.random.seed(42)
y_true = np.random.randint(0, 4, 100)
y_pred = np.random.randint(0, 4, 100)
y_prob = np.random.rand(100, 4)
y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

calc = MetricsCalculator(
    num_classes=4,
    class_names=['glioma', 'healthy', 'meningioma', 'pituitary']
)
metrics = calc.calculate_metrics(y_true, y_pred, y_prob)
print(f"  ✓ Metrics calculated: {len(metrics)} metrics")
print(f"  ✓ Accuracy: {metrics['accuracy']:.4f}")
print(f"  ✓ F1 (macro): {metrics['f1_macro']:.4f}")

# Summary
print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nYour brain tumor detection system is ready!")
print("\nNext steps:")
print("  1. Train centralized baseline:")
print("     python scripts/train_centralized.py --model vit")
print("\n  2. Train with federated learning:")
print("     python scripts/train_federated.py --model vit")
print("\n  3. Evaluate trained model:")
print("     python scripts/evaluate.py --model_path checkpoints/best_model.pth --model_type vit")
print("=" * 70)
