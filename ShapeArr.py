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
def generate_array_and_find_indexes(shape, low=0, high=100):
     """Generates an array of random integers and returns indexes of even and odd numbers.

    Args:
        shape: A tuple specifying the shape of the array (e.g., (3, 4)).
        low: The minimum integer value in the array (inclusive, default is 0).
        high: The maximum integer value in the array (exclusive, default is 100).

    Returns:
        A tuple containing:
            - The generated array.
            - A list of tuples representing the indexes of even numbers.
            - A list of tuples representing the indexes of odd numbers.

    Raises:
        ValueError: If the shape is not valid or high <= low.
    """     
     if high <= low:
        raise ValueError("High must be greater than low.")
     if not isinstance(shape, tuple) or not all(isinstance(x, int) and x > 0 for x in shape):
        raise ValueError("Shape must be a tuple of positive integers.")

     # Generate the random integer array
     array = np.random.randint(low, high, size=shape)

     # Find indexes of even and odd numbers
     even_indexes = list(zip(*np.where(array % 2 == 0)))
     odd_indexes = list(zip(*np.where(array % 2 != 0)))

     return array, even_indexes, odd_indexes
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
def test_should_return_correct_indexes_for_generated_array():
    """Tests the function for a small array."""
    shape = (2, 3)
    array, even_indexes, odd_indexes = generate_array_and_find_indexes(shape, low=1, high=10)

    for idx in even_indexes:
        assert array[idx] % 2 == 0, f"Test failed: Index {idx} is not even."
    for idx in odd_indexes:
        assert array[idx] % 2 != 0, f"Test failed: Index {idx} is not odd."

    assert len(even_indexes) + len(odd_indexes) == array.size