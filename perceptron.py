#Aqsa Noreen
import numpy as np

def perceptron_learning_algorithm(X, Y):

    num_examples, num_features = X.shape

    weights = np.zeros(num_features + 1)

    # Learning rate
    alpha = 0.1

    # Bias unit
    bias = -0.1

    # Add bias to input data
    X = np.hstack((np.ones((num_examples, 1)) * bias, X))

    # Iterate over the data multiple times
    for _ in range(1000):  # 1000 iterations as an example
        for i in range(num_examples):
            # Calculate the output (dot product between weights and input)
            output = np.dot(X[i], weights)

            # Apply step function
            prediction = 1 if output > 0 else 0

            # Update weights if there is an error
            if Y[i] != prediction:
                error = Y[i] - prediction
                weights += alpha * error * X[i]

    return weights

# Training data
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
print("Final weights:", final_weights)

