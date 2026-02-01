"""
Federated Learning Client

Implements client-side training for federated learning using Flower framework.
Each client trains on local data and sends model updates to the server.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from typing import Dict, List, Tuple
from collections import OrderedDict


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated learning.
    
    Handles local training, evaluation, and model weight exchange with the server.
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 5,
        learning_rate: float = 0.0001,
        optimizer_name: str = "adamw"
    ):
        """
        Initialize Flower client.
        
        Args:
            client_id: Unique identifier for this client
            model: PyTorch model
            train_loader: Training data loader (local data)
            val_loader: Validation data loader
            device: Device to train on
            local_epochs: Number of epochs to train locally per round
            learning_rate: Learning rate for local training
            optimizer_name: Optimizer to use (adam, adamw, sgd)
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        if optimizer_name.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.0001
            )
        elif optimizer_name.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=0.0001
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            List of numpy arrays (model weights)
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """
        Set model parameters from numpy arrays.
        
        Args:
            parameters: List of numpy arrays (model weights)
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train model on local data.
        
        Args:
            parameters: Current global model parameters
            config: Configuration for this round
            
        Returns:
            (updated_parameters, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Train for local_epochs
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (self.local_epochs * len(self.train_loader))
        
        # Get updated parameters
        updated_parameters = self.get_parameters({})
        
        # Return results
        num_examples = len(self.train_loader.dataset)
        metrics = {
            "loss": avg_loss,
            "client_id": float(self.client_id)
        }
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model on local validation data.
        
        Args:
            parameters: Current global model parameters
            config: Configuration for evaluation
            
        Returns:
            (loss, num_examples, metrics)
        """
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        metrics = {
            "accuracy": accuracy,
            "client_id": float(self.client_id)
        }
        
        return avg_loss, total, metrics


def create_flower_client(
    client_id: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict
) -> FlowerClient:
    """
    Factory function to create a Flower client.
    
    Args:
        client_id: Client identifier
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        
    Returns:
        FlowerClient instance
    """
    device = torch.device(
        config['hardware']['device'] if torch.cuda.is_available() else 'cpu'
    )
    
    return FlowerClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        local_epochs=config['federated']['local_epochs'],
        learning_rate=config['training']['learning_rate'],
        optimizer_name=config['training']['optimizer']
    )


if __name__ == "__main__":
    import numpy as np
    
    print("Federated Client Module")
    print("Use this module to create and run federated learning clients.")
    print("\nExample usage:")
    print("""
    client = FlowerClient(
        client_id=0,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cuda'),
        local_epochs=5
    )
    
    # Start client
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )
    """)
