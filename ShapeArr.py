import numpy as np
def test_should_generate_values_within_distribution():
    """Tests that the generated values follow the specified mean and standard deviation."""
    shape = (1000,)
    mean = 5
    std_dev = 2
    result = generate_normal_array(shape, mean, std_dev)
    assert np.abs(np.mean(result) - mean) < 0.1, f"Test failed: Mean deviates significantly from {mean}."
    assert np.abs(np.std(result) - std_dev) < 0.1, f"Test failed: Std deviation deviates significantly from {std_dev}."
def test_should_generate_values_within_distribution():
    shape = (1000,)
    mean = 5
    std_dev = 2
    result = generate_normal_array(shape, mean, std_dev)
    assert np.abs(np.mean(result) - mean) < 0.1, f"Test failed: Mean deviates significantly from {mean}."
    assert np.abs(np.std(result) - std_dev) < 0.1, f"Test failed: Std deviation deviates significantly from {std_dev}."
