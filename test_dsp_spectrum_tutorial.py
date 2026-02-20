"""Minimal tests for dsp_spectrum_tutorial.py.

Run:
    python test_dsp_spectrum_tutorial.py
"""

import numpy as np

from dsp_spectrum_tutorial import (
    alias_frequency,
    fft_spectrum_1d,
    gaussian_kernel_2d,
    conv2_fft,
    haar_dwt_1level,
    make_time_axis,
    synth_signal,
)


def test_alias_formula():
    fs = 1024.0
    assert abs(alias_frequency(650.0, fs) - 374.0) < 1e-9


def test_fft_peak_location():
    fs = 1024.0
    t = make_time_axis(fs, 1.0)
    x = synth_signal(t, ((123.0, 1.0),))
    s = fft_spectrum_1d(x, fs, window="hann")
    peak_f = s.freq_hz[np.argmax(s.magnitude)]
    assert abs(peak_f - 123.0) <= 1.0


def test_hann_reduces_far_leakage():
    fs = 1024.0
    t = make_time_axis(fs, 1.0)
    x = synth_signal(t, ((50.3, 1.0),))

    rect = fft_spectrum_1d(x, fs, window=None)
    hann = fft_spectrum_1d(x, fs, window="hann")

    freq = rect.freq_hz
    far = np.abs(freq - 50.3) > 30
    assert np.mean(hann.magnitude[far]) < np.mean(rect.magnitude[far])


def test_gaussian_smoothing_reduces_hf_energy():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1, (128, 128))
    k = gaussian_kernel_2d(13, 2.0)
    img_s = conv2_fft(img, k)

    gx = np.diff(img, axis=1)
    gx_s = np.diff(img_s, axis=1)
    assert np.mean(gx_s**2) < np.mean(gx**2)


def test_haar_energy_split():
    fs = 1024.0
    t = make_time_axis(fs, 1.0)
    x = synth_signal(t, ((20.0, 1.0), (200.0, 0.3)))
    a, d = haar_dwt_1level(x)
    assert np.mean(a**2) > 0
    assert np.mean(d**2) > 0


def main():
    test_alias_formula()
    test_fft_peak_location()
    test_hann_reduces_far_leakage()
    test_gaussian_smoothing_reduces_hf_energy()
    test_haar_energy_split()
    print("All dsp_spectrum_tutorial tests passed.")


if __name__ == "__main__":
    main()
