"""
Loss functions for SyncTalk 2D training.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights


class PerceptualLoss:
    """VGG-based perceptual loss for image quality."""
    
    def __init__(self, loss_fn=nn.MSELoss()):
        """
        Initialize perceptual loss.
        
        Args:
            loss_fn: Base loss function to use (default: MSELoss)
        """
        self.criterion = loss_fn
        self.contentFunc = self._build_vgg_layers()

    def _build_vgg_layers(self):
        """Build VGG feature extractor up to conv3_3."""
        conv_3_3_layer = 14
        cnn = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        
        for i, layer in enumerate(cnn):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
                
        return model

    def get_loss(self, fake_img, real_img):
        """
        Calculate perceptual loss between fake and real images.
        
        Args:
            fake_img: Generated image
            real_img: Target image
            
        Returns:
            Perceptual loss value
        """
        f_fake = self.contentFunc(fake_img)
        f_real = self.contentFunc(real_img)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


def cosine_loss(audio_embedding, visual_embedding, labels):
    """
    Calculate cosine similarity loss for audio-visual synchronization.
    
    Args:
        audio_embedding: Audio feature embeddings
        visual_embedding: Visual feature embeddings  
        labels: Target labels (1 for synchronized, 0 for not)
        
    Returns:
        BCE loss based on cosine similarity
    """
    logloss = nn.BCELoss()
    d = nn.functional.cosine_similarity(audio_embedding, visual_embedding)
    # Clamp cosine similarity to valid range [0, 1] for BCE loss
    # Cosine similarity is in [-1, 1], so we transform to [0, 1]
    d = (d + 1) / 2  # Transform from [-1, 1] to [0, 1]
    d = torch.clamp(d, min=1e-7, max=1-1e-7)  # Avoid exact 0 or 1 for numerical stability
    loss = logloss(d.unsqueeze(1), labels)
    return loss