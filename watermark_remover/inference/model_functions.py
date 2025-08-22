"""Model definitions and utilities for the Watermark Remover project.

This module defines two convolutional neural networks used throughout
the project: a U‑Net for watermark removal and a VDSR network for
super‑resolution.  It also provides helpers to convert between PIL
images and PyTorch tensors and to load the most recent or best
checkpoint from a directory of saved model weights.  A combined loss
function composed of SSIM, L1 and perceptual losses is defined at the
end of the file for training scenarios.

The implementations here mirror those used in the original research
papers but have been simplified for clarity.  They operate on single
channel (grayscale) images sized to 792×612 pixels.  All model
operations clamp outputs into the [0, 1] range to produce valid
images.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19
from torchvision.models.vgg import VGG19_Weights
from pytorch_msssim import SSIM

# Lock to guard model loading and file I/O.  When multiple threads are
# creating models or reading from disk concurrently this prevents
# occasional race conditions.
model_lock = threading.Lock()


class VDSR(nn.Module):
    """Very Deep Super‑Resolution network for image upscaling.

    This implementation uses 20 convolutional layers arranged into a
    simple residual architecture.  It accepts a single‑channel input
    and outputs a single‑channel image of the same size, learning to
    predict the residual between the low‑resolution input and the
    high‑resolution target.
    """

    def __init__(self) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        # Initial convolution
        layers.append(nn.Conv2d(1, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(64))
        # Middle layers with skip connections.  We build 9 blocks of
        # two convolutions each, which results in 18 convolutional
        # layers total in this section.  A larger number of blocks
        # typically yields better super‑resolution quality at the cost
        # of slower inference.
        for _ in range(9):
            layers.append(self.make_block(64, 64))
        # Final convolution
        layers.append(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Construct a residual block used in the VDSR network."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layers(x)
        out = out + residual
        # Clamp the output to [0, 1] to ensure valid image values
        return out.clamp(0, 1)


class UNet(nn.Module):
    """U‑Net architecture for removing watermarks from grayscale images."""

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 512)
        # Middle
        self.middle = nn.Sequential(
            self.conv_block(512, 1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
        )
        # Decoder
        self.dec5 = self.conv_block(512 + 512, 512)
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)
        # Final layer
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Two‑convolution block with batch normalisation and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        enc5 = self.enc5(F.max_pool2d(enc4, 2))
        # Middle
        middle = self.middle(F.max_pool2d(enc5, 2))
        # Align shapes for skip connection
        enc5_aligned = enc5[:, :, : middle.shape[2], :]
        middle = middle + enc5_aligned  # skip connection
        # Decoder with skip connections
        dec5 = self.dec5(torch.cat([F.interpolate(middle, size=enc5.shape[2:]), enc5], dim=1))
        dec4 = self.dec4(torch.cat([F.interpolate(dec5, size=enc4.shape[2:]), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:]), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:]), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:]), enc1], dim=1))
        final_output = self.final_conv(dec1)
        return final_output.clamp(0, 1)


def PIL_to_tensor(path: str) -> torch.Tensor:
    """Load an image from disk and convert it to a normalised tensor.

    Images are resized to 792×612 pixels and converted to grayscale.  The
    resulting tensor has shape (1, H, W) and values in [0, 1].
    """
    with model_lock:
        image = Image.open(path).convert("L")
    transform = transforms.Compose(
        [transforms.Resize((792, 612)), transforms.ToTensor()]
    )
    image_tensor = transform(image)
    return image_tensor


def tensor_to_PIL(tensor: torch.Tensor) -> Image.Image:
    """Convert a single‑image tensor back to a PIL Image."""
    array = tensor.squeeze().squeeze().cpu().numpy()
    image = Image.fromarray((array * 255).astype("uint8"), "L")
    return image


def load_best_model(model: nn.Module, directory: str) -> None:
    """Load the checkpoint with the lowest validation loss from ``directory``.

    If the directory contains no checkpoints or the files are malformed,
    this function prints a warning and returns without modifying the
    model.  When multiple checkpoints are present they are ordered
    lexicographically on the assumption that the filename contains the
    epoch number (e.g. ``model_epoch_10.pth``).
    """
    # Gather candidate checkpoint files
    with model_lock:
        model_files = [f for f in os.listdir(directory) if f.endswith(".pth")]
        model_files.sort(key=lambda f: int(f.split("_")[2].split(".")[0]) if "_" in f else 0)
    if not model_files:
        print(f"No model files found in {directory}")
        return
    recent_model_path = os.path.join(directory, model_files[-1])
    with model_lock:
        save_dict = torch.load(recent_model_path)
    val_losses: List[float] = save_dict.get("val_loss", [])
    if not val_losses:
        print(f"No validation loss values found in {recent_model_path}")
        return
    lowest_val_loss_epoch = val_losses.index(min(val_losses)) + 1
    best_model_file = f"model_epoch_{lowest_val_loss_epoch}.pth"
    best_model_path = os.path.join(directory, best_model_file)
    with model_lock:
        save_dict = torch.load(best_model_path)
    # Remove 'module.' or '_orig_mod.' prefixes from keys for DataParallel
    new_state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in 
save_dict["state_dict"].items()
    }
    model.load_state_dict(new_state_dict)
    print(
        f"Model from epoch {lowest_val_loss_epoch} loaded from {best_model_path} with validation loss {min(val_losses)}"
    )


def load_model(model: nn.Module, model_path: str) -> None:
    """Load a specific checkpoint file into ``model``.

    If the file does not exist or is malformed, this function prints
    warnings but does not raise errors.
    """
    if not os.path.isfile(model_path):
        print(f"No model file found at {model_path}")
        return
    with model_lock:
        save_dict = torch.load(model_path)
    val_loss = save_dict.get("val_loss")
    if val_loss is None:
        print(f"No validation loss value found in {model_path}")
        return
    new_state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in 
save_dict["state_dict"].items()
    }
    model.load_state_dict(new_state_dict)
    print(f"Model loaded from {model_path}")


class PerceptualLoss(nn.Module):
    """Compute the perceptual difference between two images using VGG19."""

    def __init__(self) -> None:
        super().__init__()
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = F.l1_loss(x_vgg, y_vgg)
        return loss


class CombinedLoss(nn.Module):
    """Combined SSIM, L1 and perceptual loss for training models."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 0.5) -> None:
        super().__init__()
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        ssim_loss = 1 - self.ssim_module(outputs, original)
        l1 = self.l1_loss(outputs, original)
        # Convert grayscale to 3‑channel images for perceptual loss
        outputs_3ch = torch.cat([outputs] * 3, dim=1)
        original_3ch = torch.cat([original] * 3, dim=1)
        perceptual = self.perceptual_loss(outputs_3ch, original_3ch)
        loss = self.alpha * l1 + self.beta * perceptual + self.gamma * ssim_loss
        return loss