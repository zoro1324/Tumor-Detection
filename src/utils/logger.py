"""
Utility Functions

Helper functions for logging, checkpointing, and general utilities.
"""

import os
import torch
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class Logger:
    """Simple logger for training progress."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.log(f"Logger initialized: {experiment_name}")
        self.log(f"Log file: {self.log_file}")
    
    def log(self, message: str, print_to_console: bool = True):
        """
        Log a message.
        
        Args:
            message: Message to log
            print_to_console: Whether to also print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
        if print_to_console:
            print(log_message)


class CheckpointManager:
    """Manages model checkpointing."""
    
    def __init__(self, checkpoint_dir: str, save_best_only: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Whether to only save the best model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_metric = float('-inf')
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        filename: str = "checkpoint.pth",
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if not self.save_best_only or is_best:
            filepath = self.checkpoint_dir / filename
            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved: {filepath}")
            
            if is_best:
                best_filepath = self.checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, best_filepath)
                print(f"Best model saved: {best_filepath}")
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        filename: str = "checkpoint.pth"
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            filename: Checkpoint filename
            
        Returns:
            Dictionary with checkpoint information
        """
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (accuracy), 'min' for metrics to minimize (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered after {self.counter} epochs without improvement.")
                print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        
        return False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved: {filepath}")


def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


if __name__ == "__main__":
    # Test utilities
    print("Testing Utilities...")
    
    # Test logger
    logger = Logger("logs/test", "test_experiment")
    logger.log("Test log message")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='max')
    
    scores = [0.80, 0.82, 0.81, 0.83, 0.82, 0.81, 0.80]
    for epoch, score in enumerate(scores):
        should_stop = early_stopping(score, epoch)
        print(f"Epoch {epoch}: score={score:.2f}, stop={should_stop}")
        if should_stop:
            break
    
    print("\n✓ Utilities test passed!")
