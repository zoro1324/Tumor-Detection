"""
Data Preprocessing Module for Brain Tumor MRI Images

This module handles:
- Image loading and validation
- CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement
- Normalization and standardization
- Conversion from grayscale to RGB (for pretrained models)
"""

import os
from typing import Tuple, Optional, List
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class MRIPreprocessor:
    """
    Preprocessor for MRI brain scans.
    
    Applies CLAHE enhancement, resizing, normalization, and conversion to RGB format
    suitable for pretrained vision transformers.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize the MRI preprocessor.
        
        Args:
            image_size: Target image size (width and height)
            apply_clahe: Whether to apply CLAHE enhancement
            clahe_clip_limit: Threshold for contrast limiting
            clahe_tile_grid_size: Size of grid for histogram equalization
            normalize_mean: Mean values for normalization (ImageNet stats)
            normalize_std: Std values for normalization (ImageNet stats)
        """
        self.image_size = image_size
        self.apply_clahe = apply_clahe
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Initialize CLAHE
        if apply_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_grid_size
            )
        
        # Define base transforms (no augmentation - just preprocessing)
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from path and convert to numpy array.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Grayscale image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        return image
    
    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        This enhances contrast in MRI images, making subtle features more visible.
        
        Args:
            image: Grayscale image as numpy array
            
        Returns:
            Enhanced image
        """
        return self.clahe.apply(image)
    
    def grayscale_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert grayscale image to 3-channel RGB by replicating channels.
        
        This is necessary for pretrained models expecting RGB input.
        
        Args:
            image: Grayscale image
            
        Returns:
            RGB image (3 channels with identical values)
        """
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Complete preprocessing pipeline for a single image.
        
        Steps:
        1. Load grayscale image
        2. Apply CLAHE enhancement (if enabled)
        3. Convert to RGB
        4. Resize to target size
        5. Normalize using ImageNet statistics
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image tensor (C, H, W)
        """
        # Load grayscale image
        image = self.load_image(image_path)
        
        # Apply CLAHE enhancement
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)
        
        # Convert to RGB
        image = self.grayscale_to_rgb(image)
        
        # Convert to PIL Image for torchvision transforms
        image_pil = Image.fromarray(image)
        
        # Apply transforms (resize, to tensor, normalize)
        image_tensor = self.base_transform(image_pil)
        
        return image_tensor
    
    def preprocess_batch(self, image_paths: List[str]) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batch tensor (B, C, H, W)
        """
        tensors = [self.preprocess(path) for path in image_paths]
        return torch.stack(tensors)


def validate_dataset(dataset_path: str, classes: List[str]) -> dict:
    """
    Validate dataset structure and count images per class.
    
    Args:
        dataset_path: Path to dataset root directory
        classes: List of class names (folder names)
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "total_images": 0,
        "class_distribution": {},
        "errors": []
    }
    
    dataset_path = Path(dataset_path)
    
    # Check if dataset path exists
    if not dataset_path.exists():
        results["valid"] = False
        results["errors"].append(f"Dataset path does not exist: {dataset_path}")
        return results
    
    # Check each class folder
    for class_name in classes:
        class_path = dataset_path / class_name
        
        if not class_path.exists():
            results["valid"] = False
            results["errors"].append(f"Class folder missing: {class_name}")
            continue
        
        # Count images in class folder
        image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
        num_images = len(image_files)
        
        results["class_distribution"][class_name] = num_images
        results["total_images"] += num_images
        
        if num_images == 0:
            results["valid"] = False
            results["errors"].append(f"No images found in class: {class_name}")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    preprocessor = MRIPreprocessor(
        image_size=224,
        apply_clahe=True
    )
    
    # Test on a sample image
    test_image_path = "dataset/glioma/0000.jpg"
    
    if os.path.exists(test_image_path):
        print("Testing preprocessor...")
        tensor = preprocessor.preprocess(test_image_path)
        print(f"Output tensor shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        print("✓ Preprocessing test passed!")
    else:
        print(f"Test image not found: {test_image_path}")
    
    # Validate dataset
    print("\nValidating dataset...")
    validation = validate_dataset(
        dataset_path="dataset",
        classes=["glioma", "healthy", "meningioma", "pituitary"]
    )
    
    print(f"Dataset valid: {validation['valid']}")
    print(f"Total images: {validation['total_images']}")
    print("Class distribution:")
    for class_name, count in validation["class_distribution"].items():
        print(f"  {class_name}: {count}")
    
    if validation["errors"]:
        print("Errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
