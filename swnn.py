import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.special import expit  # for stable sigmoid

rng = np.random.default_rng(42)  # Set a seed for reproducibility

def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x)

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - x**2

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0)

class AdaptiveLearningRate:
    def __init__(self, initial_rate: float = 0.01, decay: float = 1e-4, min_rate: float = 1e-5):
        self.initial_rate = initial_rate
        self.rate = initial_rate
        self.decay = decay
        self.min_rate = min_rate
        self.iterations = 0

    def get_rate(self) -> float:
        self.iterations += 1
        self.rate = max(self.initial_rate / (1 + self.decay * self.iterations), self.min_rate)
        return self.rate

class SynapseWeightedNeuron:
    def __init__(self, input_size: int, activation_function: str = 'relu'):
        self.weights = rng.standard_normal(input_size) * np.sqrt(2. / input_size)  # He initialization
        self.bias = 0.0
        self.set_activation_function(activation_function)
        self.firing_history = []
        self.max_history_length = 100
        self.adaptation_rate = 0.001  # Reduced adaptation rate

    def set_activation_function(self, activation_function: str):
        if activation_function == 'sigmoid':
            self.activate = sigmoid
            self.activate_derivative = sigmoid_derivative
        elif activation_function == 'tanh':
            self.activate = tanh
            self.activate_derivative = tanh_derivative
        elif activation_function == 'relu':
            self.activate = relu
            self.activate_derivative = relu_derivative
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, inputs: np.ndarray) -> float:
        self.last_input = inputs
        self.weighted_sum = np.dot(self.weights, inputs) + self.bias
        self.activation = self.activate(self.weighted_sum)
        self.firing_history.append(self.activation)
        if len(self.firing_history) > self.max_history_length:
            self.firing_history.pop(0)
        return self.activation

    def backward(self, error: float, learning_rate: float) -> np.ndarray:
        recent_activity = np.mean(self.firing_history[-10:])
        adaptation_factor = 1 + self.adaptation_rate * (recent_activity - 0.5)  # Center around 0.5
        delta = error * self.activate_derivative(self.activation)
        
        self.weights += learning_rate * delta * self.last_input * adaptation_factor
        self.bias += learning_rate * delta * adaptation_factor
        
        return self.weights * delta

class SynapseWeightedLayer:
    def __init__(self, input_size: int, output_size: int, activation_function: str = 'relu'):
        self.neurons = [SynapseWeightedNeuron(input_size, activation_function) for _ in range(output_size)]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def backward(self, errors: np.ndarray, learning_rate: float) -> np.ndarray:
        return np.array([neuron.backward(error, learning_rate) for neuron, error in zip(self.neurons, errors)])

class SynapseWeightedNeuralNetwork:
    def __init__(self, layer_sizes: List[int], activation_functions: List[str]):
        self.layers = [
            SynapseWeightedLayer(layer_sizes[i], layer_sizes[i+1], activation_functions[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.adaptive_lr = AdaptiveLearningRate(initial_rate=0.01, decay=1e-4, min_rate=1e-5)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, inputs: np.ndarray, target: np.ndarray) -> float:
        layer_inputs = [inputs]
        layer_outputs = []
        
        # Forward pass
        for layer in self.layers:
            layer_outputs.append(layer.forward(layer_inputs[-1]))
            layer_inputs.append(layer_outputs[-1])
        
        # Backward pass
        learning_rate = self.adaptive_lr.get_rate()
        layer_errors = [target - layer_outputs[-1]]
        for i in reversed(range(len(self.layers))):
            layer_error = self.layers[i].backward(layer_errors[-1], learning_rate)
            if i > 0:
                layer_errors.append(np.sum(layer_error, axis=0))
        
        return np.mean(np.abs(layer_errors[0]))  # Return mean absolute error

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 32) -> List[float]:
        errors = []
        n_batches = len(X) // batch_size
        for epoch in range(epochs):
            epoch_errors = []
            for i in range(n_batches):
                batch_X = X[i*batch_size:(i+1)*batch_size]
                batch_y = y[i*batch_size:(i+1)*batch_size]
                batch_errors = []
                for inputs, target in zip(batch_X, batch_y):
                    self.forward(inputs)
                    error = self.backward(inputs, target)
                    batch_errors.append(error)
                epoch_errors.append(np.mean(batch_errors))
            avg_error = np.mean(epoch_errors)
            errors.append(avg_error)
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Error: {avg_error:.6f}, Learning Rate: {self.adaptive_lr.rate:.6f}")
        return errors