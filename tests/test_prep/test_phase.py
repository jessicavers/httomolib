import time
import numpy as np
import pytest
from httomolib.prep.phase import paganin_filter
from numpy.testing import assert_allclose

eps = 1e-6

def test_paganin_filter(host_data):
    # --- testing the Paganin filter from TomoPy on tomo_standard ---#
    filtered_data = paganin_filter(host_data)

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), -1.227684e-09, rtol=eps)
    assert_allclose(np.max(filtered_data), -7.713906e-10, rtol=eps)

    #: make sure the output is float32
    assert filtered_data.dtype == np.float32


def test_paganin_filter_energy100(host_data):
    filtered_data = paganin_filter(host_data, energy=100.0)

    assert_allclose(np.mean(filtered_data), -6.506681e-10, rtol=1e-05)
    assert_allclose(np.min(filtered_data), -6.939478e-10, rtol=eps)

    assert filtered_data.ndim == 3
    assert filtered_data.dtype == np.float32


def test_paganin_filter_dist75(host_data):
    filtered_data = paganin_filter(host_data, dist=75.0, alpha=1e-6)
    
    assert_allclose(np.sum(np.mean(filtered_data, axis=(1, 2))), -2.2097976e-07, rtol=1e-6)
    assert_allclose(np.sum(filtered_data), -0.0045256666, rtol=1e-6)
    assert_allclose(np.mean(filtered_data[0, 60:63, 90]), -1.1674713e-09, rtol=1e-6)
    assert_allclose(np.sum(filtered_data[50:100, 40, 1]), -6.416675e-08, rtol=1e-6)    
