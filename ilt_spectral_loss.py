"""IL T spectral loss utilities with physical-normalized weighting.

This module is dependency-free (stdlib only) so examples/tests run even in
minimal environments. It is designed for small pedagogical tensors.
"""

from __future__ import annotations

import cmath
import math
from typing import List, Sequence, Tuple

Matrix = List[List[float]]


def _validate_same_shape(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> Tuple[int, int]:
    if len(a) == 0 or len(a[0]) == 0:
        raise ValueError("empty matrix")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("shape mismatch")
    return len(a), len(a[0])


def dft2_power(image: Sequence[Sequence[float]]) -> Matrix:
    """Return 2D DFT power |F(u,v)|^2 using a direct DFT (O(H^2W^2))."""
    h = len(image)
    w = len(image[0])
    out: Matrix = [[0.0 for _ in range(w)] for _ in range(h)]
    for u in range(h):
        for v in range(w):
            acc = 0j
            for x in range(h):
                for y in range(w):
                    phase = -2.0 * math.pi * ((u * x / h) + (v * y / w))
                    acc += image[x][y] * cmath.exp(1j * phase)
            out[u][v] = (acc.real * acc.real + acc.imag * acc.imag)
    return out


def fftshift2(m: Sequence[Sequence[float]]) -> Matrix:
    h = len(m)
    w = len(m[0])
    out: Matrix = [[0.0 for _ in range(w)] for _ in range(h)]
    h2 = h // 2
    w2 = w // 2
    for i in range(h):
        for j in range(w):
            out[(i + h2) % h][(j + w2) % w] = float(m[i][j])
    return out


def normalized_log_spectrum(power: Sequence[Sequence[float]], floor_ratio: float = 1e-6) -> Matrix:
    """Compute log spectrum in a numerically stable, physically normalized way.

    floor_ratio applies relative flooring against the spectrum max to avoid
    epsilon-dependent huge negative values dominating gradients.
    """
    h = len(power)
    w = len(power[0])
    pmax = max(max(row) for row in power)
    floor = max(pmax * floor_ratio, 1e-30)
    out: Matrix = [[0.0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            out[i][j] = math.log(max(power[i][j], floor))
    return out


def radial_frequency_map(shape: Tuple[int, int]) -> Matrix:
    h, w = shape
    cy = h // 2
    cx = w // 2
    rmax = math.sqrt(cy * cy + cx * cx) + 1e-12
    out: Matrix = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r = math.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            out[y][x] = r / rmax
    return out


def physical_normalized_weight(
    target_power_shifted: Sequence[Sequence[float]],
    alpha: float = 0.05,
    gamma: float = 0.7,
    hf_boost: float = 0.5,
) -> Matrix:
    """Build a practical weight matrix for ILT spectral loss.

    w(u,v) = (alpha + P_norm(u,v))^-gamma * (1 + hf_boost * r_norm(u,v))
    where P_norm is target spectrum normalized by global max.
    """
    h = len(target_power_shifted)
    w = len(target_power_shifted[0])
    pmax = max(max(row) for row in target_power_shifted)
    pmax = max(pmax, 1e-30)
    rmap = radial_frequency_map((h, w))

    out: Matrix = [[0.0 for _ in range(w)] for _ in range(h)]
    for i in range(h):
        for j in range(w):
            pnorm = target_power_shifted[i][j] / pmax
            base = (alpha + pnorm) ** (-gamma)
            out[i][j] = base * (1.0 + hf_boost * rmap[i][j])
    return out


def spectral_loss(
    pred: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    weight: Sequence[Sequence[float]] | None = None,
    floor_ratio: float = 1e-6,
) -> float:
    """Weighted MSE between normalized log power spectra."""
    h, w = _validate_same_shape(pred, target)

    p_pred = fftshift2(dft2_power(pred))
    p_tgt = fftshift2(dft2_power(target))
    s_pred = normalized_log_spectrum(p_pred, floor_ratio=floor_ratio)
    s_tgt = normalized_log_spectrum(p_tgt, floor_ratio=floor_ratio)

    if weight is None:
        weight = [[1.0 for _ in range(w)] for _ in range(h)]

    loss = 0.0
    total_w = 0.0
    for i in range(h):
        for j in range(w):
            wij = float(weight[i][j])
            diff = s_pred[i][j] - s_tgt[i][j]
            loss += wij * diff * diff
            total_w += wij
    return loss / max(total_w, 1e-30)


def build_demo_case(size: int = 16) -> Tuple[Matrix, Matrix]:
    """Return (target, pred) binary masks; pred has rounded line-end details."""
    target = [[0.0 for _ in range(size)] for _ in range(size)]
    pred = [[0.0 for _ in range(size)] for _ in range(size)]

    # target: T-like feature with sharper line-end corners
    for y in range(4, 12):
        target[y][7] = 1.0
    for x in range(4, 12):
        target[4][x] = 1.0

    # pred: blurred/shortened ends (typical CD/line-end miss)
    for y in range(5, 11):
        pred[y][7] = 1.0
    for x in range(5, 11):
        pred[5][x] = 1.0

    return target, pred


def demo_metrics() -> dict:
    target, pred = build_demo_case(16)
    tgt_pow = fftshift2(dft2_power(target))
    w = physical_normalized_weight(tgt_pow, alpha=0.05, gamma=0.7, hf_boost=0.8)

    plain = spectral_loss(pred, target, weight=None, floor_ratio=1e-6)
    weighted = spectral_loss(pred, target, weight=w, floor_ratio=1e-6)

    return {
        "plain_spectral_loss": plain,
        "weighted_spectral_loss": weighted,
        "dc_weight": w[len(w) // 2][len(w[0]) // 2],
        "corner_weight": w[0][0],
    }


def main() -> None:
    m = demo_metrics()
    print("=== ILT Spectral Loss Demo ===")
    for k, v in m.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
