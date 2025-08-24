import numpy as np
import matplotlib.pyplot as plt

# Activation fn
def sigmoid(z):        # used in Output layer as we want Probabilities
    return 1 / (1 + np.exp(-z))      

def sigmoid_derivative(a):
    return a * (1-a)      # Input is already activated value 'a', not raw z

def tanh(z):        # squashes value in range (-1, 1)
    return np.tanh(z) # used in hidden layers as it helps with XOR non-linearity

def tanh_derivative(a):
    return 1 - a**2

# Neural Network class
class NeuralNetwork:
    def __init__(self, layers, lr=0.5, seed=42, hidden_activation='tanh'):
        np.random.seed(seed)     # fixes the randomness
        self.lr = lr
        self.layers = layers     # storing list of no. of Neurons in each layer
        self.hidden_activation = hidden_activation  # tanh or sigmoid
        self.params = {}          # dictionary that stores all weights and biases
        self._init_weights()     # for initializing parameters


    def _init_weights(self):
        for i in range(1, len(self.layers)):
            n_in, n_out = self.layers[i-1], self.layers[i]
            limit = np.sqrt(6 / (n_in + n_out))
            self.params[f"W{i}"] = np.random.uniform(-limit, limit, (n_out, n_in))
            self.params[f"b{i}"] = np.zeros((n_out, 1))

    # Activation chooser
    def _act(self, Z, last=False):
        if last:
            return sigmoid(Z)

        return tanh(Z) if self.hidden_activation == 'tanh' else sigmoid(Z)

    def _act_derivative(self, A, last=False):
        if last:
            return sigmoid_derivative(A)
        return tanh_derivative(A) if self.hidden_activation == 'tanh' else sigmoid_derivative(A)
    # Returning correct derivative depending on whether it's hidden or output layer

    # Forward  Propagation
    def forward(self, X):
        cache = {"A0": X}
        L = len(self.layers) - 1
        for i in range(1, L+1):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]
            Z = np.dot(W, cache[f"A{i-1}"]) + b
            A = self._act(Z, last = (i==L))
            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = A

        return cache    # storing values in cache is essential for BackPropagation


    # Loss fn
    def compute_loss(self, Y, Y_hat):
        m = Y.shape[1]
        return -(1/m) * np.sum(Y*np.log(Y_hat+ 1e-12) + (1-Y)*np.log(1-Y_hat+1e-12))
        #  Binary cross-entropy loss (BCE)
        # +1e-12 to prevent from log(0)

    #  Backward  Propagation
    def backward(self, cache, Y):
        grads = {}        # Empty dict. to store gradients
        L = len(self.layers) - 1
        m = Y.shape[1]

        # dA for output layer (sigmoid + BCE)
        A_L = cache[f"A{L}"]    
        dA = -(np.divide(Y, A_L+1e-12) - np.divide(1-Y, 1-A_L + 1e-12))

        for i in reversed(range(1, L+1)):   # Iterating Backward from Last to first hidden layer
            dZ = dA * self._act_derivative(cache[f"A{i}"], last=(i==L))
            grads[f"dW{i}"] = (1/m)*np.dot(dZ, cache[f"A{i-1}"].T)
            grads[f"db{i}"] = (1/m)*np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.params[f"W{i}"].T, dZ)

        return grads

    # Update parameters
    def update_params(self, grads):
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]
            # W = W - lr*dW

    # Training loop
    def fit(self, X, Y, epochs=10000):
        train_losses, train_accs = [], []
        for epoch in range(epochs):
            cache = self.forward(X)
            A_L = cache[f"A{len(self.layers)-1}"]
            loss = self.compute_loss(Y, A_L)
            grads = self.backward(cache, Y)
            self.update_params(grads)

            train_losses.append(loss)
            preds = (A_L>0.5).astype(int)
            train_accs.append(np.mean(preds==Y))

        return np.array(train_losses), np.array(train_accs)


    #  Prediction
    def predict(self, X):
        cache = self.forward(X)
        Y_hat = cache[f"A{len(self.layers)-1}"]
        return (Y_hat > 0.5).astype(int)

# XOR Dataset- if no. of 1's = odd, output = 1, if even, output = 0
X = np.array([[0,0,1,1], [0,1,0,1]])
Y = np.array([[0,1,1,0]])

# Training
nn = NeuralNetwork([2,4,1], lr=0.5, seed=2, hidden_activation='tanh')
train_losses, train_accs = nn.fit(X, Y, epochs=10000)
#  Create Neural network with architecture: 2 input, 4 hidden neurons in 1 hidden layer, 1 output

# Prediction
print("Final Predictions: ", nn.predict(X))

fig = plt.figure(figsize=(12,5))

# Training Loss
plt.subplot(1,2,1)
plt.plot(np.arange(1, len(train_losses)+1), train_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")

# Training Accuracy
plt.subplot(1,2,2)
plt.plot(np.arange(1, len(train_accs)+1), train_accs)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")

plt.tight_layout()
plt.show()




