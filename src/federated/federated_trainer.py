"""
Federated Training Orchestrator

Coordinates federated learning training across multiple simulated clients.
Supports both IID and non-IID data distributions.
"""

import torch
import yaml
import flwr as fl
from pathlib import Path
from typing import Dict, List
import multiprocessing as mp

from ..models.model_factory import create_model_from_config
from ..data.federated_data_loader import FederatedDataManager
from .federated_client import create_flower_client
from .federated_server import create_evaluate_fn, WeightedFedAvg


class FederatedTrainer:
    """
    Orchestrates federated learning training.
    
    Manages server and multiple clients for simulated federated learning.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize federated trainer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.fed_config = self.config['federated']
        self.training_config = self.config['training']
        
        # Setup device
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"Using device: {self.device}")
    
    def train_federated(self, save_path: str = "checkpoints/federated_model.pth"):
        """
        Run federated learning training.
        
        Args:
            save_path: Path to save the final global model
        """
        print("\n" + "="*60)
        print("FEDERATED LEARNING TRAINING")
        print("="*60)
        
        # Load data
        print("\nLoading and partitioning data...")
        data_manager = FederatedDataManager(self.config_path)
        
        client_loaders, val_loader, test_loader = data_manager.get_federated_loaders(
            batch_size=self.training_config['batch_size']
        )
        
        num_clients = len(client_loaders)
        print(f"Number of clients: {num_clients}")
        
        # Create global model
        print("\nInitializing global model...")
        global_model = create_model_from_config(self.config)
        global_model.to(self.device)
        
        params_info = global_model.count_parameters()
        print(f"Model: {self.config['model']['architecture']}")
        print(f"Total parameters: {params_info['total']:,}")
        print(f"Trainable parameters: {params_info['trainable']:,}")
        
        # Create evaluation function for server
        evaluate_fn = create_evaluate_fn(global_model, test_loader, self.device)
        
        # Create federated learning strategy
        strategy = WeightedFedAvg(
            fraction_fit=self.fed_config['fraction_fit'],
            fraction_evaluate=self.fed_config['fraction_evaluate'],
            min_fit_clients=self.fed_config['min_fit_clients'],
            min_evaluate_clients=self.fed_config.get('min_evaluate_clients', num_clients),
            min_available_clients=self.fed_config['min_available_clients'],
            evaluate_fn=evaluate_fn,
        )
        
        # Get initial parameters
        initial_parameters = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        
        print(f"\nStarting federated learning for {self.fed_config['num_rounds']} rounds...")
        print(f"Local epochs per round: {self.fed_config['local_epochs']}")
        print(f"Batch size: {self.training_config['batch_size']}")
        print(f"Learning rate: {self.training_config['learning_rate']}")
        
        # Define client function
        def client_fn(cid: str):
            """Create a client instance."""
            client_id = int(cid)
            
            # Create client model (copy of global model)
            client_model = create_model_from_config(self.config)
            
            # Get client's data loader
            train_loader = client_loaders[client_id]
            
            # Create and return client
            return create_flower_client(
                client_id=client_id,
                model=client_model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=self.config
            )
        
        # Start simulation
        print("\nRunning federated learning simulation...")
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=self.fed_config['num_rounds']),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.25 if torch.cuda.is_available() else 0},
        )
        
        print("\n" + "="*60)
        print("FEDERATED LEARNING COMPLETED")
        print("="*60)
        
        # Print summary
        if history.metrics_centralized:
            print("\nFinal Results:")
            final_round = max(history.metrics_centralized['accuracy'])
            final_accuracy = history.metrics_centralized['accuracy'][final_round[-1]]
            print(f"Final Test Accuracy: {final_accuracy:.4f}")
        
        # Save final model
        print(f"\nSaving final model to: {save_path}")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_model.state_dict(), save_path)
        
        return history


if __name__ == "__main__":
    # Example usage
    print("Federated Trainer Module")
    print("\nUsage:")
    print("""
    trainer = FederatedTrainer("configs/config.yaml")
    history = trainer.train_federated(save_path="checkpoints/federated_model.pth")
    """)
