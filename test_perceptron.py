import numpy as np
import pytest
from perceptron import perceptron_learning_algorithm

def test_weights_shape():
    # Simple test data
    # X = np.array([[1, 0], [0, 1]])
    # Y = np.array([1, 0])
    X = np.array([
        [1, 0, 1, 0, 0, 0],  # A1
        [1, 0, 1, 1, 0, 0],  # A2
        [1, 0, 1, 0, 1, 0],  # A3
        [1, 1, 0, 0, 1, 1],  # A4
        [1, 1, 1, 1, 0, 0],  # A5
        [1, 0, 0, 0, 1, 1],  # A6
        [1, 0, 0, 0, 1, 0],  # A7
        [0, 1, 1, 1, 0, 1],  # A8
        [0, 1, 1, 0, 1, 1],  # A9
        [0, 0, 0, 1, 1, 0],  # A10
        [0, 1, 0, 1, 0, 1],  # A11
        [0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 1, 0, 1, 1],  # A13
        [0, 1, 1, 1, 0, 0]  # A14
    ])

    Y = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    # Train the perceptron
    final_weights = perceptron_learning_algorithm(X, Y)

    assert len(final_weights) == X.shape[1] + 1

def test_perceptron_learning():
    # Test data
    # X = np.array([
    #     [1, 0, 1, 0, 0, 0],
    #     [0, 1, 0, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1]
    # ])
    # Y = np.array([1, 0, 0])
    X = np.array([
        [1, 0, 1, 0, 0, 0],  # A1
        [1, 0, 1, 1, 0, 0],  # A2
        [1, 0, 1, 0, 1, 0],  # A3
        [1, 1, 0, 0, 1, 1],  # A4
        [1, 1, 1, 1, 0, 0],  # A5
        [1, 0, 0, 0, 1, 1],  # A6
        [1, 0, 0, 0, 1, 0],  # A7
        [0, 1, 1, 1, 0, 1],  # A8
        [0, 1, 1, 0, 1, 1],  # A9
        [0, 0, 0, 1, 1, 0],  # A10
        [0, 1, 0, 1, 0, 1],  # A11
        [0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 1, 0, 1, 1],  # A13
        [0, 1, 1, 1, 0, 0]  # A14
    ])

    Y = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

    # Train the perceptron
    final_weights = perceptron_learning_algorithm(X, Y)

    # Add bias to input data
    bias = -0.1
    X = np.hstack((np.ones((X.shape[0], 1)) * bias, X))

    # Test the classification accuracy
    correct_classifications = 0
    for i in range(X.shape[0]):
        output = np.dot(X[i], final_weights)
        prediction = 1 if output > 0 else 0
        if prediction == Y[i]:
            correct_classifications += 1

    # Check if all examples are correctly classified
    assert correct_classifications == X.shape[0]
