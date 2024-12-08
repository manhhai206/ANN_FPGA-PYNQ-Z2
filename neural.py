from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward_propagation(self, input):
        pass

    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        pass


class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5
    
    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        dweights = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * dweights
        self.bias -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        return np.dot(output_error, self.weights.T)


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def setup_loss(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        result = []
        for sample in input:
            output = sample
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, learning_rate, epochs):
        for i in range(epochs):
            err = 0
            for x, y in zip(x_train, y_train):
                output = x
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y, output)
                error = self.loss_prime(y, output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            err /= len(x_train)
            print("Epoch %d/%d, error = %f" % (i + 1, epochs, err))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def loss(y_true, y_pred):
    return 0.5 * np.sum((y_pred - y_true) ** 2)


def loss_prime(y_true, y_pred):
    return y_pred - y_true


x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[1]]])
x_test = np.array([[[1, 1]], [[1, 1]], [[0, 0]], [[0, 1]]])
net = Network()
net.add(FCLayer((1, 2), (1, 3)))
net.add(ActivationLayer(sigmoid, sigmoid_prime))  
net.add(FCLayer((1, 3), (1, 3)))  
net.add(ActivationLayer(sigmoid, sigmoid_prime)) 
net.add(FCLayer((1, 3), (1, 1)))
net.add(ActivationLayer(sigmoid, sigmoid_prime)) 

net.setup_loss(loss, loss_prime)

net.fit(x_train, y_train, epochs=1000, learning_rate=0.5)

out = net.predict(x_test)
print(out)

