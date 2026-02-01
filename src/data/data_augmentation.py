"""
Data Augmentation Module for Medical Imaging

Implements medical imaging-specific augmentations using Albumentations.
Carefully designed to preserve diagnostic features while improving model robustness.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Any


class MedicalImageAugmentation:
    """
    Augmentation pipeline for medical MRI images.
    
    Uses conservative augmentations suitable for brain tumor detection:
    - Geometric: rotation, flip, affine
    - Intensity: brightness/contrast, Gaussian noise
    
    Avoids aggressive augmentations that could alter diagnostic features.
    """
    
    def __init__(self, config: Dict[str, Any], mode: str = "train"):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration dictionary
            mode: One of 'train', 'val', 'test'
        """
        self.mode = mode
        self.config = config
        self.transform = self._build_transform()
    
    def _build_transform(self) -> A.Compose:
        """Build the augmentation pipeline based on mode."""
        
        if self.mode == "train":
            return self._build_train_transform()
        elif self.mode == "val":
            return self._build_val_transform()
        else:  # test
            return self._build_test_transform()
    
    def _build_train_transform(self) -> A.Compose:
        """
        Build training augmentation pipeline.
        
        Conservative augmentations for medical imaging:
        - Horizontal/vertical flips
        - Small rotations (±15°)
        - Minor shift/scale/rotate
        - Brightness/contrast adjustment
        - Gaussian noise (simulates acquisition noise)
        """
        cfg = self.config.get("train", {})
        
        transforms_list = []
        
        # Geometric augmentations
        if "horizontal_flip" in cfg:
            transforms_list.append(
                A.HorizontalFlip(p=cfg["horizontal_flip"])
            )
        
        if "vertical_flip" in cfg:
            transforms_list.append(
                A.VerticalFlip(p=cfg["vertical_flip"])
            )
        
        if "rotation_limit" in cfg:
            transforms_list.append(
                A.Rotate(
                    limit=cfg["rotation_limit"],
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.7
                )
            )
        
        if "shift_scale_rotate" in cfg:
            ssr = cfg["shift_scale_rotate"]
            transforms_list.append(
                A.ShiftScaleRotate(
                    shift_limit=ssr.get("shift_limit", 0.1),
                    scale_limit=ssr.get("scale_limit", 0.1),
                    rotate_limit=ssr.get("rotate_limit", 15),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=ssr.get("p", 0.5)
                )
            )
        
        # Intensity augmentations
        if "brightness_contrast" in cfg:
            bc = cfg["brightness_contrast"]
            transforms_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=bc.get("brightness_limit", 0.2),
                    contrast_limit=bc.get("contrast_limit", 0.2),
                    p=bc.get("p", 0.5)
                )
            )
        
        if "gaussian_noise" in cfg:
            gn = cfg["gaussian_noise"]
            transforms_list.append(
                A.GaussNoise(
                    var_limit=tuple(gn.get("var_limit", [10.0, 50.0])),
                    mean=0,
                    p=gn.get("p", 0.3)
                )
            )
        
        return A.Compose(transforms_list)
    
    def _build_val_transform(self) -> A.Compose:
        """
        Build validation augmentation pipeline.
        
        Minimal or no augmentation for validation.
        """
        return A.Compose([])  # No augmentation for validation
    
    def _build_test_transform(self) -> A.Compose:
        """
        Build test augmentation pipeline.
        
        No augmentation for test set.
        """
        return A.Compose([])  # No augmentation for test
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to an image.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            
        Returns:
            Augmented image
        """
        # Albumentations expects (H, W, C) format
        if len(image.shape) == 2:
            # Add channel dimension for grayscale
            image = np.expand_dims(image, axis=-1)
        
        augmented = self.transform(image=image)
        return augmented["image"]


def get_augmentation_pipeline(config: Dict[str, Any], mode: str = "train") -> MedicalImageAugmentation:
    """
    Factory function to create augmentation pipeline.
    
    Args:
        config: Full configuration dictionary with 'augmentation' key
        mode: One of 'train', 'val', 'test'
        
    Returns:
        MedicalImageAugmentation instance
    """
    aug_config = config.get("augmentation", {})
    return MedicalImageAugmentation(aug_config, mode=mode)


# Example conservative augmentation preset for medical imaging
MEDICAL_IMAGING_PRESET = {
    "train": {
        "horizontal_flip": 0.5,
        "vertical_flip": 0.3,
        "rotation_limit": 15,
        "shift_scale_rotate": {
            "shift_limit": 0.1,
            "scale_limit": 0.1,
            "rotate_limit": 15,
            "p": 0.5
        },
        "brightness_contrast": {
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "p": 0.5
        },
        "gaussian_noise": {
            "var_limit": [10.0, 50.0],
            "p": 0.3
        }
    },
    "val": {
        "enabled": False
    },
    "test": {
        "enabled": False
    }
}


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Test with medical imaging preset
    print("Testing Medical Image Augmentation...")
    
    aug = MedicalImageAugmentation(MEDICAL_IMAGING_PRESET, mode="train")
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    print(f"Input image shape: {dummy_image.shape}")
    
    # Apply augmentation
    augmented = aug(dummy_image)
    
    print(f"Augmented image shape: {augmented.shape}")
    print("✓ Augmentation test passed!")
    
    # Test with different modes
    for mode in ["train", "val", "test"]:
        aug = MedicalImageAugmentation(MEDICAL_IMAGING_PRESET, mode=mode)
        result = aug(dummy_image)
        print(f"Mode '{mode}': {result.shape}")
