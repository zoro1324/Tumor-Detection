"""
Model Evaluation Script

Evaluate trained models and generate comprehensive reports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import yaml
from pathlib import Path

from src.models.model_factory import create_model_from_config
from src.data.federated_data_loader import FederatedDataManager
from src.evaluation.metrics import evaluate_model, MetricsCalculator
from src.utils.logger import save_metrics


def evaluate(config_path: str, model_path: str, model_type: str = 'vit'):
    """
    Evaluate a trained model.
    
    Args:
        config_path: Path to configuration file
        model_path: Path to model checkpoint
        model_type: Model type ('vit' or 'swin')
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model architecture
    if model_type == 'vit' and 'swin' in config['model']['architecture']:
        config['model']['architecture'] = 'vit_base_patch16_224'
    elif model_type == 'swin' and 'vit' in config['model']['architecture']:
        config['model']['architecture'] = 'swin_tiny_patch4_window7_224'
    
    print("\n" + "="*70)
    print(f"MODEL EVALUATION - {Path(model_path).name}")
    print("="*70)
    
    # Setup device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    data_manager = FederatedDataManager(config_path)
    _, _, test_loader = data_manager.get_centralized_loaders(
        batch_size=config['training']['batch_size']
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = create_model_from_config(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = evaluate_model(
        model, test_loader, device,
        num_classes=config['data']['num_classes'],
        class_names=config['data']['classes']
    )
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    if 'auc_roc_macro' in metrics:
        print(f"  AUC-ROC (macro): {metrics['auc_roc_macro']:.4f}")
        print(f"  AUC-ROC (weighted): {metrics['auc_roc_weighted']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for class_name in config['data']['classes']:
        print(f"\n  {class_name.capitalize()}:")
        print(f"    Precision: {metrics[f'precision_{class_name}']:.4f}")
        print(f"    Recall: {metrics[f'recall_{class_name}']:.4f}")
        print(f"    F1-Score: {metrics[f'f1_{class_name}']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print("       ", "  ".join([f"{cls[:4]:>4}" for cls in config['data']['classes']]))
    for i, row in enumerate(cm):
        print(f"{config['data']['classes'][i][:4]:>4}:  ", "  ".join([f"{val:>4}" for val in row]))
    
    # Save results
    model_name = Path(model_path).stem
    results_path = Path(config['paths']['results_dir']) / f'{model_name}_evaluation.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(results_path))
    
    print(f"\n✓ Evaluation completed! Results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate brain tumor classifier")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['vit', 'swin'], default='vit',
                        help='Model type')
    
    args = parser.parse_args()
    
    evaluate(args.config, args.model_path, args.model_type)
