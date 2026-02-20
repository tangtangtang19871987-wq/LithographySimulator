"""DSP 入门实验脚本：FFT/采样/窗函数/滤波/图像频谱与常见坑。

运行：
    python dsp_spectrum_tutorial.py

该脚本只依赖 NumPy，可选 Matplotlib（用于可视化）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


EPS = 1e-12


@dataclass
class Spectrum1D:
    freq_hz: np.ndarray
    magnitude: np.ndarray
    phase: np.ndarray


def make_time_axis(fs: float, duration_s: float) -> np.ndarray:
    n = int(round(fs * duration_s))
    return np.arange(n) / fs


def synth_signal(t: np.ndarray, tones: Tuple[Tuple[float, float], ...], dc: float = 0.0) -> np.ndarray:
    x = np.full_like(t, dc, dtype=np.float64)
    for f_hz, amp in tones:
        x += amp * np.sin(2 * np.pi * f_hz * t)
    return x


def fft_spectrum_1d(x: np.ndarray, fs: float, window: str | None = None, remove_mean: bool = True) -> Spectrum1D:
    x = np.asarray(x, dtype=np.float64)
    if remove_mean:
        x = x - np.mean(x)

    if window is None:
        w = np.ones_like(x)
    elif window == "hann":
        w = np.hanning(len(x))
    elif window == "hamming":
        w = np.hamming(len(x))
    else:
        raise ValueError(f"unknown window={window}")

    coherent_gain = np.mean(w)
    xw = x * w

    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(len(x), d=1 / fs)

    mag = np.abs(X) * 2 / (len(x) * max(coherent_gain, EPS))
    if len(mag) > 0:
        mag[0] *= 0.5
    phase = np.angle(X)
    return Spectrum1D(freq_hz=f, magnitude=mag, phase=phase)


def alias_frequency(f_hz: float, fs: float) -> float:
    return abs(((f_hz + fs / 2) % fs) - fs / 2)


def mix_down(x: np.ndarray, fs: float, lo_hz: float) -> np.ndarray:
    t = np.arange(len(x)) / fs
    return x * np.cos(2 * np.pi * lo_hz * t)


def lowpass_mask(freq: np.ndarray, cutoff_hz: float) -> np.ndarray:
    return (np.abs(freq) <= cutoff_hz).astype(np.float64)


def bandpass_mask(freq: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return ((np.abs(freq) >= low_hz) & (np.abs(freq) <= high_hz)).astype(np.float64)


def freq_filter_1d(x: np.ndarray, fs: float, mask_builder) -> np.ndarray:
    X = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), d=1 / fs)
    mask = mask_builder(freq)
    y = np.fft.ifft(X * mask)
    return np.real(y)


def gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("size must be odd")
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= np.sum(k)
    return k


def conv2_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = image.shape
    kh, kw = kernel.shape
    H = h + kh - 1
    W = w + kw - 1

    F = np.fft.fft2(image, s=(H, W))
    K = np.fft.fft2(kernel, s=(H, W))
    y = np.fft.ifft2(F * K).real

    r0 = (kh - 1) // 2
    c0 = (kw - 1) // 2
    return y[r0:r0 + h, c0:c0 + w]


def fft_magnitude_image(img: np.ndarray, log_scale: bool = True) -> np.ndarray:
    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    if log_scale:
        mag = np.log1p(mag)
    return mag


def haar_dwt_1level(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) % 2 != 0:
        x = x[:-1]
    even = x[0::2]
    odd = x[1::2]
    approx = (even + odd) / math.sqrt(2)
    detail = (even - odd) / math.sqrt(2)
    return approx, detail


def synthetic_image(size: int = 128) -> np.ndarray:
    y, x = np.indices((size, size))
    img = np.zeros((size, size), dtype=np.float64)
    img += ((x // 8) % 2) * 0.6
    img += ((y // 16) % 2) * 0.3
    img += np.exp(-((x - size * 0.65) ** 2 + (y - size * 0.35) ** 2) / (2 * (size * 0.06) ** 2))
    return img


def run_all_demos() -> dict:
    fs = 1024.0
    t = make_time_axis(fs, 1.0)

    x_clean = synth_signal(t, ((50, 1.0), (120, 0.5)))
    x_offgrid = synth_signal(t, ((50.3, 1.0), (120, 0.5)))

    spec_rect = fft_spectrum_1d(x_offgrid, fs, window=None)
    spec_hann = fft_spectrum_1d(x_offgrid, fs, window="hann")

    x_alias = synth_signal(t, ((650, 1.0),))
    spec_alias = fft_spectrum_1d(x_alias, fs)

    x_mixed = mix_down(x_clean, fs, lo_hz=100)
    spec_mixed = fft_spectrum_1d(x_mixed, fs, window="hann")

    y_lp = freq_filter_1d(x_clean, fs, lambda f: lowpass_mask(f, 80))
    y_bp = freq_filter_1d(x_clean, fs, lambda f: bandpass_mask(f, 100, 130))

    img = synthetic_image(128)
    gk = gaussian_kernel_2d(13, 2.2)
    img_s = conv2_fft(img, gk)

    spec_img = fft_magnitude_image(img)
    spec_img_s = fft_magnitude_image(img_s)

    approx, detail = haar_dwt_1level(x_clean)

    results = {
        "alias_expect": alias_frequency(650, fs),
        "alias_peak": float(spec_alias.freq_hz[np.argmax(spec_alias.magnitude)]),
        "leakage_rect_peak": float(np.max(spec_rect.magnitude)),
        "leakage_hann_peak": float(np.max(spec_hann.magnitude)),
        "mixed_main_peak": float(spec_mixed.freq_hz[np.argmax(spec_mixed.magnitude)]),
        "lp_energy": float(np.mean(y_lp**2)),
        "bp_energy": float(np.mean(y_bp**2)),
        "img_hf_before": float(np.mean(spec_img[50:78, 50:78])),
        "img_hf_after": float(np.mean(spec_img_s[50:78, 50:78])),
        "haar_energy_ratio": float(np.mean(detail**2) / max(np.mean(approx**2), EPS)),
    }
    return results


def main() -> None:
    r = run_all_demos()
    print("=== DSP Tutorial Demo Summary ===")
    for k, v in r.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
