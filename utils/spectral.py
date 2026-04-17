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


def fda_transfer(source, target, beta):
    """FDA: swap low-freq amplitude of source with target's.

    Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation" (CVPR 2020).
    Preserves source phase (structure/content), adopts target amplitude (style/color).

    Args:
        source: (B, C, H, W) source images in [-1, 1]
        target: (B, C, H, W) target images in [-1, 1] (same or different B)
        beta: float, fraction of frequency radius for low-freq mask

    Returns:
        adapted: (B, C, H, W) source with target low-freq amplitude
    """
    B, C, H, W = source.shape

    # FFT
    src_freq = torch.fft.fft2(source, dim=(-2, -1))
    tgt_freq = torch.fft.fft2(target, dim=(-2, -1))

    src_freq_shifted = torch.fft.fftshift(src_freq, dim=(-2, -1))
    tgt_freq_shifted = torch.fft.fftshift(tgt_freq, dim=(-2, -1))

    # Amplitude and phase
    src_amp = src_freq_shifted.abs()
    src_phase = src_freq_shifted.angle()
    tgt_amp = tgt_freq_shifted.abs()

    # Low-freq mask
    cy, cx = H // 2, W // 2
    radius = min(H, W) * beta / 2.0
    y = torch.arange(H, device=source.device).float()
    x = torch.arange(W, device=source.device).float()
    Y, X = torch.meshgrid(y, x, indexing="ij")
    dist = ((Y - cy) ** 2 + (X - cx) ** 2).sqrt()
    mask = (dist <= radius).float().unsqueeze(0).unsqueeze(0)

    # Swap low-freq amplitude
    mixed_amp = src_amp * (1 - mask) + tgt_amp * mask

    # Reconstruct with mixed amplitude + source phase
    mixed_freq = mixed_amp * torch.exp(1j * src_phase)
    mixed_freq = torch.fft.ifftshift(mixed_freq, dim=(-2, -1))
    adapted = torch.fft.ifft2(mixed_freq, dim=(-2, -1)).real

    return adapted.clamp(-1, 1)
