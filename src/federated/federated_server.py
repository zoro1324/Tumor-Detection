"""
Federated Learning Server

Implements the central server for federated learning using Flower framework.
Handles model aggregation and global evaluation.
"""

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.server.strategy import FedAvg
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class FederatedServer:
    """
    Central server for federated learning.
    
    Coordinates training across multiple clients and aggregates model updates
    using FedAvg (Federated Averaging) strategy.
    """
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[callable] = None,
        on_fit_config_fn: Optional[callable] = None,
        on_evaluate_config_fn: Optional[callable] = None,
    ):
        """
        Initialize federated server.
        
        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum number of clients that must connect
            evaluate_fn: Function for centralized evaluation on server
            on_fit_config_fn: Function to configure client training
            on_evaluate_config_fn: Function to configure client evaluation
        """
        self.strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
        )
    
    def start(
        self,
        server_address: str = "0.0.0.0:8080",
        num_rounds: int = 100,
        config: Optional[Dict[str, Scalar]] = None
    ):
        """
        Start the federated learning server.
        
        Args:
            server_address: Address to bind the server
            num_rounds: Number of federated learning rounds
            config: Additional configuration parameters
        """
        # Start Flower server
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=self.strategy,
        )


class WeightedFedAvg(FedAvg):
    """
    Custom FedAvg strategy with weighted averaging based on client dataset sizes.
    
    Gives more weight to clients with larger datasets during aggregation.
    """
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights from multiple clients.
        
        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failures
            
        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}
        
        # Extract weights and number of examples
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Perform weighted averaging
        weights_prime = aggregate_weighted(weights_results)
        
        # Convert back to Parameters
        parameters_aggregated = ndarrays_to_parameters(weights_prime)
        
        # Aggregate custom metrics
        metrics_aggregated = {}
        if len(results) > 0:
            # Calculate weighted average of metrics
            total_examples = sum([fit_res.num_examples for _, fit_res in results])
            
            # Aggregate accuracy
            if any(['accuracy' in fit_res.metrics for _, fit_res in results]):
                accuracy_aggregated = sum([
                    fit_res.metrics.get('accuracy', 0) * fit_res.num_examples
                    for _, fit_res in results
                ]) / total_examples
                metrics_aggregated['accuracy'] = accuracy_aggregated
            
            # Aggregate loss
            if any(['loss' in fit_res.metrics for _, fit_res in results]):
                loss_aggregated = sum([
                    fit_res.metrics.get('loss', 0) * fit_res.num_examples
                    for _, fit_res in results
                ]) / total_examples
                metrics_aggregated['loss'] = loss_aggregated
        
        return parameters_aggregated, metrics_aggregated


def aggregate_weighted(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Compute weighted average of model parameters.
    
    Args:
        results: List of (weights, num_examples) tuples
        
    Returns:
        Aggregated weights
    """
    # Calculate total number of examples
    total_examples = sum([num_examples for _, num_examples in results])
    
    # Create a list of weighted arrays
    weighted_arrays = [
        [layer * num_examples for layer in weights]
        for weights, num_examples in results
    ]
    
    # Sum weighted arrays
    weights_prime = [
        np.sum([weighted_array[i] for weighted_array in weighted_arrays], axis=0) / total_examples
        for i in range(len(weighted_arrays[0]))
    ]
    
    return weights_prime


def create_evaluate_fn(model, test_loader, device):
    """
    Create evaluation function for centralized server-side evaluation.
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Evaluation function
    """
    def evaluate(
        server_round: int,
        parameters: Parameters,
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate global model on centralized test set.
        
        Returns:
            (loss, metrics) tuple
        """
        import torch
        
        # Set model parameters
        params_dict = zip(model.state_dict().keys(), parameters_to_ndarrays(parameters))
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, {"accuracy": accuracy}
    
    return evaluate


if __name__ == "__main__":
    print("Federated Server Module")
    print("Use this module to create and start a federated learning server.")
    print("\nExample usage:")
    print("""
    server = FederatedServer(
        fraction_fit=1.0,
        min_fit_clients=4,
        min_available_clients=4
    )
    server.start(num_rounds=100)
    """)
