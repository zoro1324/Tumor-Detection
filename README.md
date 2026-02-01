# Brain Tumor Detection with ViT/Swin Transformer and Federated Learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Advanced brain tumor detection from MRI scans using **Vision Transformer (ViT)** and **Swin Transformer** models with **Federated Learning** for privacy-preserving collaborative training.

## 🎯 Project Objectives

1. **Develop ViT and Swin Transformer models** for accurate brain tumor detection in MRI scans
2. **Implement Federated Learning** for privacy-preserving collaborative training across healthcare centers
3. **Enhance diagnostic performance** in medical imaging using advanced AI techniques

## 📊 Dataset

- **Source**: Kaggle Brain Tumor MRI Dataset
- **Total Images**: 7,023 MRI scans
- **Classes**: 4 (Glioma, Healthy, Meningioma, Pituitary)
- **Format**: Grayscale JPG images
- **Distribution**: Well-balanced across all classes

### Class Distribution
- **Glioma**: 1,621 images (23.1%)
- **Healthy**: 2,000 images (28.5%)
- **Meningioma**: 1,645 images (23.4%)
- **Pituitary**: 1,757 images (25.0%)

## 🏗️ Architecture

### Models Implemented
1. **Vision Transformer (ViT)**
   - Architecture: ViT-Base with patch size 16×16
   - Pre-trained on ImageNet
   - ~86M parameters

2. **Swin Transformer**
   - Architecture: Swin-Tiny with hierarchical attention
   - Window size: 7×7
   - ~28M parameters

### Federated Learning
- **Framework**: Flower FL
- **Strategy**: FedAvg (Federated Averaging)
- **Clients**: 4 simulated healthcare centers
- **Distribution**: IID and non-IID data partitioning
- **Privacy**: Only model weights shared, raw data stays local

## 📁 Project Structure

```
Tumor-Detection/
├── dataset/                 # MRI image dataset
│   ├── glioma/
│   ├── healthy/
│   ├── meningioma/
│   └── pituitary/
├── src/
│   ├── data/               # Data loading and preprocessing
│   │   ├── data_preprocessing.py
│   │   ├── data_augmentation.py
│   │   └── federated_data_loader.py
│   ├── models/             # Model architectures
│   │   ├── vit_model.py
│   │   ├── swin_transformer.py
│   │   └── model_factory.py
│   ├── federated/          # Federated learning components
│   │   ├── federated_server.py
│   │   ├── federated_client.py
│   │   └── federated_trainer.py
│   ├── evaluation/         # Metrics and evaluation
│   │   └── metrics.py
│   └── utils/              # Utilities
│       └── logger.py
├── scripts/                # Training and evaluation scripts
│   ├── train_centralized.py
│   ├── train_federated.py
│   └── evaluate.py
├── configs/                # Configuration files
│   └── config.yaml
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
└── results/                # Evaluation results
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Tumor-Detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `configs/config.yaml` to customize:
- Model architecture (ViT or Swin)
- Training hyperparameters
- Federated learning settings
- Data augmentation parameters

### 3. Training

#### Centralized Training (Baseline)

```bash
# Train ViT
python scripts/train_centralized.py --model vit

# Train Swin Transformer
python scripts/train_centralized.py --model swin
```

#### Federated Learning

```bash
# Train ViT with Federated Learning
python scripts/train_federated.py --model vit

# Train Swin with Federated Learning
python scripts/train_federated.py --model swin
```

### 4. Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py --model_path checkpoints/best_model.pth --model_type vit
```

## 🔧 Key Features

### Data Preprocessing
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for MRI images
- **Normalization**: ImageNet statistics for transfer learning
- **Augmentation**: Medical imaging-specific transformations (rotation, flip, brightness/contrast adjustment)

### Training Features
- **Transfer Learning**: Pre-trained weights from ImageNet
- **Mixed Precision Training**: Automatic mixed precision (AMP) for faster training
- **Early Stopping**: Prevents overfitting with configurable patience
- **Learning Rate Scheduling**: Cosine annealing, step decay, or reduce on plateau
- **TensorBoard Logging**: Real-time training visualization

### Federated Learning
- **Data Partitioning**: IID and non-IID (Dirichlet distribution)
- **Weighted Aggregation**: FedAvg with client dataset size weighting
- **Privacy-Preserving**: Only model updates shared, never raw data
- **Multi-Client Simulation**: 4 simulated healthcare centers by default

## 📈 Expected Performance

### Centralized Training (Baseline)
- **ViT**: 92-95% accuracy
- **Swin Transformer**: 93-96% accuracy

### Federated Learning
- **ViT (Federated)**: 89-93% accuracy (2-3% drop from centralized)
- **Swin (Federated)**: 90-94% accuracy (2-3% drop from centralized)

## 📊 Evaluation Metrics

- Overall Accuracy
- Precision, Recall, F1-Score (macro and weighted)
- Per-Class Metrics
- Confusion Matrix
- AUC-ROC Curves
- Classification Report

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, torchvision, timm
- **Federated Learning**: Flower (flwr)
- **Data Processing**: NumPy, Pandas, OpenCV, Albumentations
- **Visualization**: Matplotlib, Seaborn, TensorBoard
- **Metrics**: scikit-learn

## 📝 Configuration

Key configuration parameters in `configs/config.yaml`:

```yaml
# Model
model:
  architecture: "vit_base_patch16_224"  # or "swin_tiny_patch4_window7_224"
  pretrained: true
  dropout: 0.1

# Training
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.0001
  optimizer: "adamw"
  lr_scheduler: "cosine"

# Federated Learning
federated:
  enabled: true
  num_clients: 4
  num_rounds: 100
  local_epochs: 5
  distribution_type: "iid"  # or "non_iid"
```

## 🧪 Reproducibility

For reproducible results:
1. Set `random_seed: 42` in config.yaml
2. PyTorch deterministic mode is supported
3. All random operations use seeded generators

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{brain_tumor_detection_fl,
  title={Brain Tumor Detection with ViT/Swin Transformer and Federated Learning},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/Tumor-Detection}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kaggle Brain Tumor MRI Dataset contributors
- PyTorch and timm library maintainers
- Flower Federated Learning framework
- Medical imaging research community

## 📞 Contact

For questions or feedback, please open an issue or contact [your-email@example.com]

---

**Note**: This project is for research and educational purposes. Always consult medical professionals for clinical diagnosis.
