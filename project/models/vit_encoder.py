"""
Vision Transformer (ViT) encoder for extracting visual features.

Uses a pretrained ViT model from HuggingFace to convert images
into fixed-dimensional embeddings for use in RL.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Optional
from transformers import ViTModel, ViTImageProcessor


class ViTEncoder:
    """
    Extracts visual features from images using a pretrained Vision Transformer.
    
    The encoder uses the CLS token embedding from the final layer as
    the image representation, resulting in a 768-dimensional feature vector.
    
    Attributes:
        model_name: HuggingFace model identifier
        device: Torch device for inference
        embedding_dim: Dimension of output embeddings (768 for ViT-base)
    """
    
    DEFAULT_MODEL = "google/vit-base-patch16-224"
    
    def __init__(
        self,
        model_name: str = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ViT encoder.
        
        Args:
            model_name: HuggingFace model identifier (default: google/vit-base-patch16-224)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache downloaded model weights
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading ViT encoder: {self.model_name}")
        print(f"Using device: {self.device}")
        
        # Load pretrained model and processor
        self.processor = ViTImageProcessor.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )
        
        self.model = ViTModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Freeze all weights - no training
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Store embedding dimension
        self.embedding_dim = self.model.config.hidden_size  # 768 for ViT-base
        
        print(f"ViT encoder loaded. Embedding dimension: {self.embedding_dim}")
    
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for ViT input.
        
        Handles resizing to 224x224 and normalization according to
        the model's expected input format.
        
        Args:
            image: Input image as numpy array (H, W, 3) or PIL Image
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            # Ensure uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Use HuggingFace processor for preprocessing
        # This handles resizing to 224x224 and normalization
        inputs = self.processor(images=image, return_tensors="pt")
        
        return inputs["pixel_values"].to(self.device)
    
    def encode(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract feature embedding from an image.
        
        Args:
            image: Input image as numpy array (H, W, 3) or PIL Image
            
        Returns:
            Feature embedding as numpy array of shape (embedding_dim,)
        """
        # Preprocess the image
        pixel_values = self.preprocess(image)
        
        # Forward pass through ViT (no gradient computation)
        with torch.no_grad():
            outputs = self.model(pixel_values)
        
        # Extract CLS token embedding (first token of last hidden state)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Convert to numpy and squeeze batch dimension
        embedding = cls_embedding.cpu().numpy().squeeze(0)
        
        return embedding
    
    def encode_batch(self, images: list) -> np.ndarray:
        """
        Extract feature embeddings from a batch of images.
        
        Args:
            images: List of images (numpy arrays or PIL Images)
            
        Returns:
            Feature embeddings as numpy array of shape (batch_size, embedding_dim)
        """
        # Convert all images to PIL format
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
            pil_images.append(img)
        
        # Batch preprocess
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)
        
        # Extract CLS token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return cls_embeddings.cpu().numpy()
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of output embeddings.
        
        Returns:
            Embedding dimension (768 for ViT-base)
        """
        return self.embedding_dim


def test_vit_encoder():
    """Test function to verify ViT encoder functionality."""
    print("Testing ViT Encoder...")
    
    # Create encoder
    encoder = ViTEncoder()
    
    # Create a dummy 84x84 RGB image
    dummy_image = np.random.randint(0, 256, size=(84, 84, 3), dtype=np.uint8)
    
    # Extract features
    features = encoder.encode(dummy_image)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature stats - min: {features.min():.4f}, max: {features.max():.4f}, mean: {features.mean():.4f}")
    
    # Test batch encoding
    batch = [dummy_image, dummy_image, dummy_image]
    batch_features = encoder.encode_batch(batch)
    print(f"Batch output shape: {batch_features.shape}")
    
    print("ViT Encoder test passed!")


if __name__ == "__main__":
    test_vit_encoder()
