"""FFT-based spectral decomposition utilities."""
import torch


def fft_decompose(image, beta):
    """Decompose image into low-freq and high-freq components.

    Args:
        image: (B, C, H, W) tensor
        beta: float in (0, 1), fraction of frequency radius to keep as "low"

    Returns:
        low_freq: (B, C, H, W) low-frequency spatial image
        high_freq: (B, C, H, W) high-frequency residual
    """
    B, C, H, W = image.shape

    # 2D FFT per channel
    freq = torch.fft.fft2(image, dim=(-2, -1))
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

    # Circular low-pass mask
    cy, cx = H // 2, W // 2
    radius = min(H, W) * beta / 2.0

    y = torch.arange(H, device=image.device).float()
    x = torch.arange(W, device=image.device).float()
    Y, X = torch.meshgrid(y, x, indexing="ij")
    dist = ((Y - cy) ** 2 + (X - cx) ** 2).sqrt()
    mask = (dist <= radius).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Separate
    low_shifted = freq_shifted * mask
    high_shifted = freq_shifted * (1.0 - mask)

    # IFFT back to spatial
    low_freq = torch.fft.ifft2(
        torch.fft.ifftshift(low_shifted, dim=(-2, -1)), dim=(-2, -1)
    ).real
    high_freq = torch.fft.ifft2(
        torch.fft.ifftshift(high_shifted, dim=(-2, -1)), dim=(-2, -1)
    ).real

    return low_freq, high_freq


def fft_recombine(translated_low, original_high):
    """Recombine translated low-freq with original high-freq."""
    return torch.clamp(translated_low + original_high, -1.0, 1.0)
