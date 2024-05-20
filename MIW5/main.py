import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def relu(X):
    """
    ReLU activation function.
    """
    return np.maximum(0, X)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with random weights and biases.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases for the hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        # Initialize weights and biases for the output layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        """
        Perform forward pass through the network.
        """
        # Hidden layer computation (ReLU activation)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)  # Use ReLU activation

        # Output layer computation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.z2

        return self.output

    # def forward(self, X):
    #     """
    #     Perform forward pass through the network.
    #     """
    #     # Hidden layer computation (sigmoid activation)
    #     self.z1 = np.dot(X, self.W1) + self.b1
    #     self.a1 = sigmoid(self.z1)  # Using sigmoid activation
    #
    #     # Output layer computation
    #     self.z2 = np.dot(self.a1, self.W2) + self.b2
    #     self.output = self.z2
    #
    #     return self.output

    def backward(self, X, y, learning_rate=0.01):
        """
        Perform backward propagation to update weights and biases.
        """
        # Compute gradients for the output layer
        delta2 = self.output - y
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # Compute gradients for the hidden layer
        delta1 = np.dot(delta2, self.W2.T) * (self.z1 > 0)  # Gradient for ReLU
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train_batch(self, X, y, epochs=1000, learning_rate=0.01, batch_size=10):
        """
        Train the neural network using batch gradient descent with a specified batch size.
        """
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, batch_size):
                xi_batch = X_shuffled[i:i + batch_size]
                yi_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                output = self.forward(xi_batch)

                # Backward pass
                self.backward(xi_batch, yi_batch, learning_rate)

                # Calculate mean squared error
                mse = np.mean(np.square(yi_batch - output))

                # # Print results every 100 epochs
                # if epoch % 10000 == 0 and i == 0:
                #     print(f'Epoch {epoch}, Batch {i}-{i + batch_size}, MSE: {mse}')

    def train_online(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Train the neural network using online (one example at a time) gradient descent.
        """
        n_samples = X.shape[0]
        for epoch in range(epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                xi = X_shuffled[[i], :]
                yi = y_shuffled[[i], :]

                # Forward pass
                output = self.forward(xi)

                # Backward pass
                self.backward(xi, yi, learning_rate)

                # Calculate mean squared error
                mse = np.mean(np.square(yi - output))

def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_size, epochs, learning_rate, training_method):
    """
    Train and evaluate the neural network model with given parameters.
    """
    input_size = X_train.shape[1]
    output_size = 1  # Assuming a regression task

    print(f"\nTraining Model with Hidden Size: {hidden_size}, Epochs: {epochs}, Learning Rate: {learning_rate}, Training Method: {training_method}")

    model = NeuralNetwork(input_size, hidden_size, output_size)

    if training_method == 'batch':
        model.train_batch(X_train, y_train, epochs, learning_rate)
    elif training_method == 'online':
        model.train_online(X_train, y_train, epochs, learning_rate)
    else:
        raise ValueError(f"Invalid training method: {training_method}")

    # Test the model
    predictions = model.forward(X_test)
    mse_test = np.mean(np.square(y_test - predictions))
    print(f'Final MSE on test data: {mse_test}')

# Load data
data = np.loadtxt('dane16.txt')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]



# train_and_evaluate(X_train, y_train, X_test, y_test, 10, 10000, 0.0001, 'batch')
# train_and_evaluate(X_train, y_train, X_test, y_test, 20, 20000, 0.0002, 'batch')
# train_and_evaluate(X_train, y_train, X_test, y_test, 30, 30000, 0.0003, 'batch')
# train_and_evaluate(X_train, y_train, X_test, y_test, 10, 10000, 0.0001, 'online')
# train_and_evaluate(X_train, y_train, X_test, y_test, 20, 20000, 0.0002, 'online')
# train_and_evaluate(X_train, y_train, X_test, y_test, 30, 30000, 0.0003, 'online')
train_and_evaluate(X_train, y_train, X_test, y_test, 100, 100000, 0.0001, 'online')
