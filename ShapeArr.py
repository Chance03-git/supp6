import numpy as np
def generate_normal_array(shape, mean, std_dev):
    """ Args:
        shape: A tuple specifying the shape of the array (e.g., (3, 4)).
        mean: The mean of the normal distribution.
        std_dev: The standard deviation of the normal distribution.

    Returns:
        A NumPy array with the specified shape, filled with numbers 
        from a normal distribution with the given mean and standard deviation.

    Raises:
        ValueError: If the standard deviation is negative.
    """
    if std_dev < 0:
        raise ValueError("Standard deviation cannot be negative.")

    return np.random.normal(loc=mean, scale=std_dev, size=shape)
def solve_cramers_rule(coeff_matrix, constants):
    """Solves a system of linear equations using Cramer's rule.

    Args:
        coeff_matrix: A 2D list or NumPy array representing the coefficients of the variables.
        constants: A 1D list or NumPy array representing the constants on the right-hand side.

    Returns:
        A list of solutions for the variables.

    Raises:
        ValueError: If the coefficient matrix is not square or if its determinant is zero.
    """
    coeff_matrix = np.array(coeff_matrix, dtype=float)
    constants = np.array(constants, dtype=float)

    # Check if the coefficient matrix is square
    if coeff_matrix.shape[0] != coeff_matrix.shape[1]:
        raise ValueError("Coefficient matrix must be square.")
    
    # Check if determinant of the coefficient matrix is zero
    det_main = np.linalg.det(coeff_matrix)
    if det_main == 0:
        raise ValueError("The system has no unique solution (determinant is zero).")
    
    # Solve for each variable using Cramer's Rule
    num_variables = coeff_matrix.shape[0]
    solutions = []
    
    for i in range(num_variables):
        # Create a modified matrix by replacing the i-th column with the constants
        modified_matrix = coeff_matrix.copy()
        modified_matrix[:, i] = constants
        # Calculate the determinant of the modified matrix
        det_modified = np.linalg.det(modified_matrix)
        # Calculate the solution for the i-th variable
        solutions.append(det_modified / det_main)

    return solutions


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
def test_should_raise_exception_for_negative_std_dev():

     shape = (3, 3)
     mean = 0
     std_dev = -1
     try:
        generate_normal_array(shape, mean, std_dev)
        assert False, "Test failed: Expected ValueError for negative std_dev."
     except ValueError as e:
        assert str(e) == "Standard deviation cannot be negative."
def test_should_solve_2x2_system():
    """Tests solving a simple 2x2 system."""
    coeff_matrix = [[2, -1], [1, 1]]
    constants = [1, 5]
    result = solve_cramers_rule(coeff_matrix, constants)
    assert np.allclose(result, [2, 3]), f"Test failed: Expected [2, 3], got {result}"