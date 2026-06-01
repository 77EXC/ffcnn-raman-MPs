"""
FFCNN: Fourier Feature Convolutional Neural Network
==========================================

Core module for Fourier-based spectral feature extraction.

Modified from: Frequency Convolutional Neural Network (FreqCNN)
Reference: Qu et al. "AF-2" concept for spectral analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierUnit(nn.Module):
    """
    Fourier Unit for spectral domain feature extraction.

    Performs FFT on input features and learns frequency-domain representations.
    """

    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        # Convolutions operate on real/imaginary concatenated channels
        self.conv = nn.Conv2d(
            in_channels * 2,
            out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.shape

        # Apply FFT
        fft_result = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)

        # Stack real and imaginary parts
        fft_stacked = torch.stack([fft_result.real, fft_result.imag], dim=-1)
        fft_permuted = fft_stacked.permute(0, 1, 4, 2, 3).contiguous()
        fft_flat = fft_permuted.view(batch, -1, h, w // 2 + 1)

        # Learn frequency-domain features
        fft_flat = self.conv(fft_flat)
        fft_flat = self.relu(self.bn(fft_flat))

        # Reshape back to complex
        fft_complex = fft_flat.view(batch, -1, 2, h, w // 2 + 1)
        fft_complex = fft_complex.permute(0, 1, 3, 4, 2).contiguous()
        fft_complex = torch.complex(fft_complex[..., 0], fft_complex[..., 1])

        # Inverse FFT
        output = torch.fft.irfft2(fft_complex, s=(h, w), dim=(-2, -1), norm=self.fft_norm)

        return output


class SpectralAttention(nn.Module):
    """
    Channel attention for spectral features.
    """

    def __init__(self, channels, reduction=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x.view(b, c, -1)).squeeze(-1)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class FFCNNBlock(nn.Module):
    """
    FFCNN Block: Combines Fourier transform with residual learning.
    """

    def __init__(self, in_channels, out_channels, use_attention=True):
        super(FFCNNBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Fourier unit for frequency-domain features
        self.fourier = FourierUnit(out_channels, out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Optional attention
        self.attention = SpectralAttention(out_channels) if use_attention else None

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply Fourier in 2D reshaped format
        # Reshape for Fourier unit: (B, C, H, 1)
        out_2d = out.unsqueeze(-1)
        out_2d = self.fourier(out_2d)
        out = out_2d.squeeze(-1)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        out = out + identity
        out = self.relu(out)

        if self.attention is not None:
            out = self.attention(out)

        return out


class FFCNN(nn.Module):
    """
    Fourier Feature Convolutional Neural Network for Raman spectra classification.

    Architecture:
    - Input: Raman spectra (batch, 1, length)
    - Hierarchical FFCNN blocks with Fourier features
    - Global pooling
    - Fully connected classifier

    Args:
        input_length: Number of spectral points (default: 2964)
        num_classes: Number of polymer classes (default: 10)
        hidden_channels: Base channel dimension (default: 64)
    """

    def __init__(self, input_length=2964, num_classes=10, hidden_channels=64):
        super(FFCNN, self).__init__()

        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Hierarchical FFCNN blocks
        self.block1 = FFCNNBlock(hidden_channels, hidden_channels * 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.block2 = FFCNNBlock(hidden_channels * 2, hidden_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.block3 = FFCNNBlock(hidden_channels * 2, hidden_channels * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        final_dim = hidden_channels * 4
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch, length) -> (batch, 1, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Hierarchical features
        x = self.input_conv(x)
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        output = self.classifier(x)

        return torch.sigmoid(output)

    def get_feature_maps(self, x):
        """
        Extract intermediate feature maps for visualization.

        Args:
            x: Input tensor (batch, length)

        Returns:
            List of feature maps at different stages
        """
        features = []

        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_conv(x)
        features.append(x)

        x = self.block1(x)
        x = self.pool1(x)
        features.append(x)

        x = self.block2(x)
        x = self.pool2(x)
        features.append(x)

        return features


def create_model(input_length=2964, num_classes=10, hidden_channels=64):
    """
    Factory function to create FFCNN model.

    Args:
        input_length: Number of spectral points
        num_classes: Number of output classes
        hidden_channels: Base channel dimension

    Returns:
        FFCNN model
    """
    model = FFCNN(
        input_length=input_length,
        num_classes=num_classes,
        hidden_channels=hidden_channels
    )
    return model


# =============================================================================
# Baseline Models for Comparison
# =============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN baseline for comparison."""

    def __init__(self, input_length=2964, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * (input_length // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return torch.sigmoid(x)


class ResNet1D(nn.Module):
    """ResNet-style 1D CNN baseline for comparison."""

    def __init__(self, input_length=2964, num_classes=10):
        super(ResNet1D, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.MaxPool1d(2))

        for _ in range(blocks):
            layers.append(ResidualBlock1D(in_channels, out_channels))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return torch.sigmoid(x)


class ResidualBlock1D(nn.Module):
    """Basic residual block for 1D CNN."""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out


if __name__ == '__main__':
    # Test model creation
    model = create_model(input_length=2964, num_classes=10)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(4, 2964)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")