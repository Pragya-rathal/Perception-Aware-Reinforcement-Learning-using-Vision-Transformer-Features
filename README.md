markdown


# Perception-Aware Reinforcement Learning using Vision Transformer Features
## Problem Statement
This project investigates whether structured visual representations from pretrained Vision Transformers (ViT) can improve reinforcement learning performance compared to learning directly from raw pixels. We implement and compare two RL pipelines in a custom 2D visual navigation environment:
1. **Pixel-based RL (Baseline)**: Learns directly from 84x84 RGB images using a CNN policy
2. **Feature-based RL**: Uses pretrained ViT embeddings (768-dim) as state representation with an MLP policy
## Architecture
### Environment
┌─────────────────────────────────────┐ │ 10x10 Grid World │ │ ┌───┬───┬───┬───┬───┬───┬───┬───┐ │ │ │ │ │ R │ │ │ │ │ │ │ R = Obstacle (Red) │ ├───┼───┼───┼───┼───┼───┼───┼───┤ │ G = Target (Green) │ │ │ B │ │ │ R │ │ │ │ │ B = Agent (Blue) │ ├───┼───┼───┼───┼───┼───┼───┼───┤ │ │ │ │ │ │ │ │ │ G │ │ │ │ └───┴───┴───┴───┴───┴───┴───┴───┘ │ │ │ │ Actions: 0=up, 1=down, 2=left, 3=right │ Rewards: +10 (target), -10 (obstacle), -0.1 (step) └─────────────────────────────────────┘



### Pipeline A: Pixel-Based RL
┌──────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │ 84x84x3 │───▶│ CNN Layers │───▶│ FC Layers │───▶│ Action │ │ Image │ │ (SB3 CnnPolicy)│ │ │ │ │ └──────────┘ └─────────────┘ └─────────────┘ └────────┘



### Pipeline B: Feature-Based RL
┌──────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │ 84x84x3 │───▶│ ViT Encoder │───▶│ 768-dim CLS │───▶│ Linear(128) │───▶│ Action │ │ Image │ │ (frozen) │ │ Token │ │ + MLP Policy│ │ │ └──────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └────────┘



## Project Structure
project/ ├── env/ │ ├── init.py │ ├── environment.py # Custom Gymnasium environment │ └── renderer.py # Grid-to-image renderer ├── models/ │ ├── init.py │ └── vit_encoder.py # ViT feature extractor ├── rl/ │ ├── init.py │ ├── train_pixel.py # Pixel-based training pipeline │ └── train_feature.py # Feature-based training pipeline ├── experiments/ │ ├── init.py │ ├── compare.py # Model comparison utilities │ └── plots.py # Visualization functions ├── results/ # Output directory for models and plots ├── main.py # Entry point ├── requirements.txt # Dependencies └── README.md # This file



## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
Usage
bash


# Run complete experiment pipeline
python main.py
# Or run individual components
python -m rl.train_pixel      # Train pixel-based model only
python -m rl.train_feature    # Train feature-based model only
python -m experiments.compare # Compare trained models
Results Summary
After training both models for ~3000 episodes:

Metric	Pixel-Based	Feature-Based
Final Avg Reward	~X.XX	~Y.YY
Success Rate	~XX%	~YY%
Convergence (episodes)	~XXXX	~YYYY
Note: Actual results will vary based on random seeds and hyperparameters.

Observations
Sample Efficiency: Feature-based RL typically shows faster initial learning due to pretrained visual representations capturing meaningful structure.

Final Performance: Both approaches can achieve similar final performance, but the feature-based method often converges faster.

Computational Trade-offs:

Pixel-based: Slower per-step (CNN forward pass during training)
Feature-based: ViT inference overhead, but simpler policy network
Generalization: ViT features may generalize better to visual variations since they encode semantic information learned from diverse image data.

Dependencies
Python 3.10+
PyTorch 2.0+
Stable-Baselines3
HuggingFace Transformers
Gymnasium
NumPy, Matplotlib
License
MIT License



---
## **project/requirements.txt**
torch>=2.0.0 torchvision>=0.15.0 transformers>=4.30.0 stable-baselines3>=2.0.0 gymnasium>=0.29.0 numpy>=1.24.0 matplotlib>=3.7.0 Pillow>=9.5.0 tqdm>=4.65.0



---
## **project/env/__init__.py**
```python
"""
Environment module for the visual navigation task.
"""
from env.environment import VisualNavigationEnv
from env.renderer import GridRenderer
__all__ = ["VisualNavigationEnv", "GridRenderer"]
