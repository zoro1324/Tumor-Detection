"""
Evaluation Metrics Module

Comprehensive metrics for binary and multiclass classification.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Calculate comprehensive classification metrics."""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Predicted probabilities (N, num_classes) - optional
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = per_class_precision[i]
            metrics[f'recall_{class_name}'] = per_class_recall[i]
            metrics[f'f1_{class_name}'] = per_class_f1[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # AUC-ROC (if probabilities provided)
        if y_prob is not None:
            try:
                # One-vs-rest AUC for multiclass
                metrics['auc_roc_macro'] = roc_auc_score(
                    y_true, y_prob, average='macro', multi_class='ovr'
                )
                metrics['auc_roc_weighted'] = roc_auc_score(
                    y_true, y_prob, average='weighted', multi_class='ovr'
                )
            except Exception as e:
                print(f"Warning: Could not calculate AUC-ROC: {e}")
                metrics['auc_roc_macro'] = 0.0
                metrics['auc_roc_weighted'] = 0.0
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: List[str] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run evaluation on
        num_classes: Number of  classes
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(num_classes, class_names)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_prob)
    
    # Add average loss
    metrics['loss'] = total_loss / len(data_loader.dataset)
    
    return metrics


if __name__ == "__main__":
    # Test metrics calculator
    print("Testing Metrics Calculator...")
    
    # Create dummy predictions
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 100)
    y_pred = np.random.randint(0, 4, 100)
    y_prob = np.random.rand(100, 4)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    # Calculate metrics
    calc = MetricsCalculator(
        num_classes=4,
        class_names=['glioma', 'healthy', 'meningioma', 'pituitary']
    )
    
    metrics = calc.calculate_metrics(y_true, y_pred, y_prob)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if key != 'confusion_matrix':
            print(f"  {key}: {value:.4f}")
    
    print("\nClassification Report:")
    print(calc.get_classification_report(y_true, y_pred))
    
    print("\n✓ Metrics calculator test passed!")
