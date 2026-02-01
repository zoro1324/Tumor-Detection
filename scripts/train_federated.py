"""
Federated Training Script

Train ViT or Swin Transformer models using Federated Learning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from pathlib import Path

from src.federated.federated_trainer import FederatedTrainer
from src.evaluation.metrics import evaluate_model
from src.data.federated_data_loader import FederatedDataManager
from src.models.model_factory import create_model_from_config
from src.utils.logger import save_metrics
import torch


def train_federated(config_path: str, model_type: str = "vit"):
    """
    Train model using federated learning.
    
    Args:
        config_path: Path to configuration file
        model_type: Model type ('vit' or 'swin')
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model architecture based on model_type
    if model_type == "vit":
        config['model']['architecture'] = config['model'].get('architecture', 'vit_base_patch16_224')
        if 'swin' in config['model']['architecture']:
            config['model']['architecture'] = 'vit_base_patch16_224'
    else:  # swin
        config['model']['architecture'] = config['model'].get('architecture', 'swin_tiny_patch4_window7_224')
        if 'vit' in config['model']['architecture']:
            config['model']['architecture'] = 'swin_tiny_patch4_window7_224'
    
    # Save updated config temporarily
    temp_config_path = f"configs/temp_{model_type}_federated_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create trainer
    trainer = FederatedTrainer(temp_config_path)
    
    # Train
    save_path = f"checkpoints/federated_{model_type}_model.pth"
    history = trainer.train_federated(save_path=save_path)
    
    # Final Evaluation on Test Set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load data for testing
    data_manager = FederatedDataManager(temp_config_path)
    _, _, test_loader = data_manager.get_centralized_loaders(
        batch_size=config['training']['batch_size']
    )
    
    # Load best model
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    model = create_model_from_config(config)
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    
    # Evaluate
    test_metrics = evaluate_model(
        model, test_loader, device,
        num_classes=config['data']['num_classes'],
        class_names=config['data']['classes']
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {test_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    if 'auc_roc_macro' in test_metrics:
        print(f"  AUC-ROC (macro): {test_metrics['auc_roc_macro']:.4f}")
    
    # Save results
    results_path = Path(config['paths']['results_dir']) / f'federated_{model_type}_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(test_metrics, str(results_path))
    
    # Clean up temp config
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    print(f"\n✓ Federated training completed! Model saved at: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brain tumor classifier (federated)")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['vit', 'swin'], default='vit',
                        help='Model type to train')
    
    args = parser.parse_args()
    
    train_federated(args.config, args.model)
