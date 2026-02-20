"""Tests for ilt_spectral_loss.py (stdlib only)."""

import math

from ilt_spectral_loss import (
    build_demo_case,
    dft2_power,
    fftshift2,
    normalized_log_spectrum,
    physical_normalized_weight,
    spectral_loss,
)


def test_log_spectrum_is_finite_with_flooring():
    p = [[0.0, 0.0], [0.0, 10.0]]
    s = normalized_log_spectrum(p, floor_ratio=1e-6)
    for row in s:
        for v in row:
            assert math.isfinite(v)


def test_weight_suppresses_dc_relative_to_hf():
    target, _ = build_demo_case(16)
    tp = fftshift2(dft2_power(target))
    w = physical_normalized_weight(tp, alpha=0.05, gamma=0.7, hf_boost=1.0)

    dc = w[len(w) // 2][len(w[0]) // 2]
    corner = w[0][0]
    assert corner > dc


def test_weighted_loss_highlights_line_end_error():
    target, pred = build_demo_case(16)
    tp = fftshift2(dft2_power(target))
    w = physical_normalized_weight(tp, alpha=0.05, gamma=0.7, hf_boost=1.0)

    plain = spectral_loss(pred, target, weight=None)
    weighted = spectral_loss(pred, target, weight=w)
    assert weighted > plain


def main():
    test_log_spectrum_is_finite_with_flooring()
    test_weight_suppresses_dc_relative_to_hf()
    test_weighted_loss_highlights_line_end_error()
    print("All ILT spectral loss tests passed.")


if __name__ == "__main__":
    main()
