"""
Centralized Training Script

Train ViT or Swin Transformer models in centralized manner (baseline).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import yaml

from src.models.model_factory import create_model_from_config
from src.data.federated_data_loader import FederatedDataManager
from src.evaluation.metrics import evaluate_model, MetricsCalculator
from src.utils.logger import Logger, CheckpointManager, EarlyStopping, save_metrics


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    writer: SummaryWriter = None
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Log to TensorBoard
        if writer and batch_idx % 10 == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), step)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device
):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_centralized(config_path: str, model_type: str = "vit"):
    """
    Train model in centralized manner.
    
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
    
    print("\n" + "="*70)
    print(f"CENTRALIZED TRAINING - {config['model']['architecture'].upper()}")
    print("="*70)
    
    # Setup device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading data...")
    data_manager = FederatedDataManager(config_path)
    train_loader, val_loader, test_loader = data_manager.get_centralized_loaders(
        batch_size=config['training']['batch_size']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model...")
    model = create_model_from_config(config)
    model = model.to(device)
    
    params_info = model.count_parameters()
    print(f"Total parameters: {params_info['total']:,}")
    print(f"Trainable parameters: {params_info['trainable']:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=config['training'].get('label_smoothing', 0.0))
    
    if config['training']['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    
    # Learning rate scheduler
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['lr_min']
        )
    elif config['training']['lr_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    elif config['training']['lr_scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5
        )
    else:
        scheduler = None
    
    # Utilities
    logger = Logger(config['paths']['logs_dir'], f"centralized_{model_type}")
    checkpoint_manager = CheckpointManager(
        config['paths']['checkpoints_dir'],
        save_best_only=config['logging']['save_best_only']
    )
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta'],
        mode='max'
    ) if config['training']['early_stopping']['enabled'] else None
    
    # TensorBoard
    writer = SummaryWriter(config['paths']['tensorboard_dir'] + f'/centralized_{model_type}') \
        if config['logging']['use_tensorboard'] else None
    
    # Training loop
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Optimizer: {config['training']['optimizer']}")
    
    best_val_acc = 0.0
    
    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Log
        logger.log(
            f"Epoch {epoch+1}/{config['training']['num_epochs']} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        if writer:
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Accuracy', train_acc, epoch)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Accuracy', val_acc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch,
            {'val_acc': val_acc, 'val_loss': val_loss},
            filename=f'centralized_{model_type}_epoch_{epoch+1}.pth',
            is_best=is_best
        )
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopping and early_stopping(val_acc, epoch):
            break
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    # Load best model
    checkpoint = checkpoint_manager.load_checkpoint(model, filename='best_model.pth')
    
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
    results_path = Path(config['paths']['results_dir']) / f'centralized_{model_type}_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(test_metrics, str(results_path))
    
    if writer:
        writer.close()
    
    print(f"\n✓ Training completed! Best model saved at: checkpoints/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brain tumor classifier (centralized)")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['vit', 'swin'], default='vit',
                        help='Model type to train')
    
    args = parser.parse_args()
    
    train_centralized(args.config, args.model)
