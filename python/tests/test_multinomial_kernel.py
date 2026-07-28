import numpy as np

from gasm.common.multinomial_kernel import (
    multinomial_kernel,
    multinomial_params,
    quantize_spectrum,
)


def test_quantize_spectrum_and_params_roundtrip() -> None:
    spectrum = np.array([0.5, 0.3, 0.2])
    quantized = quantize_spectrum(spectrum, quantization=10)
    np.testing.assert_array_equal(quantized, [5, 3, 2])
    params = multinomial_params(quantized, quantization=10)
    np.testing.assert_allclose(params, [0.5, 0.3, 0.2])


def test_multinomial_kernel_sums_to_one_over_support() -> None:
    quantization = 4
    params = np.array([0.5, 0.5])
    total = sum(
        multinomial_kernel(np.array([x0, quantization - x0]), params, quantization)
        for x0 in range(quantization + 1)
    )
    assert np.isclose(total, 1.0)


def test_multinomial_kernel_zero_when_support_mismatch() -> None:
    params = np.array([1.0, 0.0])
    assert multinomial_kernel(np.array([1, 1]), params, quantization=2) == 0.0
