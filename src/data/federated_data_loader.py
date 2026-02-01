"""
Federated Data Loader Module

Handles data loading and partitioning for federated learning:
- Split dataset into train/val/test
- Partition training data across multiple clients (IID and non-IID)
- Create data loaders for centralized and federated training
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import yaml

from .data_preprocessing import MRIPreprocessor
from .data_augmentation import get_augmentation_pipeline


class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor MRI images.
    
    Supports both centralized and federated learning scenarios.
    """
    
    def __init__(
        self,
        dataset_path: str,
        classes: List[str],
        preprocessor: MRIPreprocessor,
        augmentation=None,
        mode: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            dataset_path: Path to dataset root directory
            classes: List of class names
            preprocessor: MRIPreprocessor instance
            augmentation: Augmentation pipeline (optional)
            mode: One of 'train', 'val', 'test'
        """
        self.dataset_path = Path(dataset_path)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.preprocessor = preprocessor
        self.augmentation = augmentation
        self.mode = mode
        
        # Load all image paths and labels
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Load all image paths and their corresponding labels.
        
        Returns:
            List of (image_path, label) tuples
        """
        samples = []
        
        for class_name in self.classes:
            class_path = self.dataset_path / class_name
            
            if not class_path.exists():
                print(f"Warning: Class folder not found: {class_path}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            for img_file in class_path.glob("*.jpg"):
                samples.append((str(img_file), class_idx))
            
            for img_file in class_path.glob("*.png"):
                samples.append((str(img_file), class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Index
            
        Returns:
            (image_tensor, label) tuple
        """
        img_path, label = self.samples[idx]
        
        # Load and preprocess image
        # Load as numpy array first for augmentation
        image = self.preprocessor.load_image(img_path)
        
        # Apply CLAHE if enabled
        if self.preprocessor.apply_clahe:
            image = self.preprocessor.apply_clahe_enhancement(image)
        
        # Apply augmentation (if any) before converting to RGB
        if self.augmentation is not None:
            # Augmentation expects (H, W) or (H, W, C)
            augmented = self.augmentation(image)
            image = augmented
        
        # Convert to RGB
        image = self.preprocessor.grayscale_to_rgb(image)
        
        # Convert to PIL and apply final transforms
        image_pil = Image.fromarray(image)
        image_tensor = self.preprocessor.base_transform(image_pil)
        
        return image_tensor, label


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset into train, validation, and test sets.
    
    Ensures stratified split (maintains class distribution).
    
    Args:
        dataset: BrainTumorDataset instance
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        (train_subset, val_subset, test_subset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Group indices by class (stratified split)
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    train_indices, val_indices, test_indices = [], [], []
    
    # Split each class separately
    for label, indices in class_indices.items():
        random.shuffle(indices)
        
        n_samples = len(indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
    
    # Shuffle the indices
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, val_subset, test_subset


def partition_data_iid(
    dataset: Dataset,
    num_clients: int,
    random_seed: int = 42
) -> List[Subset]:
    """
    Partition dataset into IID (Independent and Identically Distributed) clients.
    
    Each client receives a similar class distribution.
    
    Args:
        dataset: Dataset to partition
        num_clients: Number of clients
        random_seed: Random seed
        
    Returns:
        List of dataset subsets, one per client
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get all indices
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    
    # Split roughly equally
    client_indices = np.array_split(all_indices, num_clients)
    
    # Create subsets
    client_datasets = [Subset(dataset, indices.tolist()) for indices in client_indices]
    
    return client_datasets


def partition_data_non_iid(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    random_seed: int = 42
) -> List[Subset]:
    """
    Partition dataset into non-IID clients using Dirichlet distribution.
    
    Creates realistic heterogeneous data distributions across clients.
    Lower alpha = more heterogeneous.
    
    Args:
        dataset: Dataset to partition
        num_clients: Number of clients
        alpha: Dirichlet distribution parameter (0.1-10.0, lower = more skewed)
        random_seed: Random seed
        
    Returns:
        List of dataset subsets, one per client
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        if isinstance(dataset, Subset):
            _, label = dataset.dataset.samples[dataset.indices[idx]]
        else:
            _, label = dataset.samples[idx]
        class_indices[label].append(idx)
    
    num_classes = len(class_indices)
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute to clients using Dirichlet distribution
    for label, indices in class_indices.items():
        random.shuffle(indices)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Split indices according to proportions
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, proportions)
        
        # Assign to clients
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split)
    
    # Shuffle each client's indices
    for indices in client_indices:
        random.shuffle(indices)
    
    # Create subsets
    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    
    return client_datasets


class FederatedDataManager:
    """
    Manager for federated learning data loading and partitioning.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize federated data manager.
        
        Args:
            config_path: Path to config.yaml file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.fed_config = self.config['federated']
        self.hardware_config = self.config['hardware']
        
        # Initialize preprocessor
        self.preprocessor = MRIPreprocessor(
            image_size=self.data_config['image_size'],
            apply_clahe=self.data_config['apply_clahe'],
            clahe_clip_limit=self.data_config['clahe_clip_limit'],
            clahe_tile_grid_size=tuple(self.data_config['clahe_tile_grid_size']),
            normalize_mean=tuple(self.data_config['normalize_mean']),
            normalize_std=tuple(self.data_config['normalize_std'])
        )
        
        # Initialize augmentation
        self.train_aug = get_augmentation_pipeline(self.config, mode='train')
        self.val_aug = get_augmentation_pipeline(self.config, mode='val')
        
        # Create full dataset (no augmentation initially)
        self.full_dataset = BrainTumorDataset(
            dataset_path=self.data_config['dataset_path'],
            classes=self.data_config['classes'],
            preprocessor=self.preprocessor,
            augmentation=None,
            mode='train'
        )
        
        # Split into train/val/test
        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(
            self.full_dataset,
            train_ratio=self.data_config['train_ratio'],
            val_ratio=self.data_config['val_ratio'],
            test_ratio=self.data_config['test_ratio'],
            random_seed=self.data_config['random_seed']
        )
        
        print(f"Dataset split: Train={len(self.train_dataset)}, "
              f"Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
    
    def get_centralized_loaders(
        self,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for centralized training.
        
        Args:
            batch_size: Batch size
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.hardware_config['num_workers'],
            pin_memory=self.hardware_config['pin_memory']
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hardware_config['num_workers'],
            pin_memory=self.hardware_config['pin_memory']
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hardware_config['num_workers'],
            pin_memory=self.hardware_config['pin_memory']
        )
        
        return train_loader, val_loader, test_loader
    
    def get_federated_loaders(
        self,
        batch_size: int
    ) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
        """
        Get data loaders for federated learning.
        
        Args:
            batch_size: Batch size
            
        Returns:
            (client_loaders, val_loader, test_loader)
        """
        num_clients = self.fed_config['num_clients']
        distribution = self.fed_config['distribution_type']
        
        # Partition training data across clients
        if distribution == 'iid':
            client_datasets = partition_data_iid(
                self.train_dataset,
                num_clients=num_clients,
                random_seed=self.data_config['random_seed']
            )
        else:  # non-iid
            alpha = self.fed_config.get('non_iid_alpha', 0.5)
            client_datasets = partition_data_non_iid(
                self.train_dataset,
                num_clients=num_clients,
                alpha=alpha,
                random_seed=self.data_config['random_seed']
            )
        
        # Create loaders for each client
        client_loaders = []
        for client_id, client_dataset in enumerate(client_datasets):
            loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.hardware_config['num_workers'],
                pin_memory=self.hardware_config['pin_memory']
            )
            client_loaders.append(loader)
            print(f"Client {client_id}: {len(client_dataset)} samples")
        
        # Validation and test loaders (centralized)
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hardware_config['num_workers'],
            pin_memory=self.hardware_config['pin_memory']
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.hardware_config['num_workers'],
            pin_memory=self.hardware_config['pin_memory']
        )
        
        return client_loaders, val_loader, test_loader


if __name__ == "__main__":
    # Test federated data manager
    print("Testing Federated Data Manager...")
    
    try:
        manager = FederatedDataManager("configs/config.yaml")
        
        # Test centralized loaders
        print("\n=== Centralized Loaders ===")
        train_loader, val_loader, test_loader = manager.get_centralized_loaders(batch_size=32)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test federated loaders
        print("\n=== Federated Loaders ===")
        client_loaders, val_loader, test_loader = manager.get_federated_loaders(batch_size=32)
        print(f"Number of clients: {len(client_loaders)}")
        
        # Test loading a batch
        print("\n=== Testing Batch Loading ===")
        images, labels = next(iter(train_loader))
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        print(f"Unique labels in batch: {torch.unique(labels).tolist()}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
