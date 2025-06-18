import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Core Components
# ---------------------------------------------------------------------------

# --- Activation Functions ---
class ActivationFunctions:
    """A collection of static methods for activation functions and their derivatives."""
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        Leaky ReLU allows a small, non-zero gradient when the unit is not active.
        Helps prevent the 'dying ReLU' problem.
        """
        return np.where(x > 0, x, x * alpha)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Derivative of the Leaky ReLU function."""
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)
class Loss:
    """Base class for all loss functions."""
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    """Mean Squared Error loss function."""
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss function, suitable for binary classification."""
    def loss(self, y_true, y_pred):
        # Clip predictions to prevent log(0) errors.
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        """Updates the layer's weights and biases using the calculated gradients."""
        layer.weights -= self.learning_rate * layer.delta_weights
        layer.bias -= self.learning_rate * layer.delta_bias

# ---------------------------------------------------------------------------
# Network Architecture
# ---------------------------------------------------------------------------

# --- Layer Class ---
class Layer:
    """Represents a single dense layer in the neural network."""
    def __init__(self, input_size, output_size, activation_name="sigmoid"):
        """Initializes the layer's weights, biases, and activation function."""
        
        self.activation_name = activation_name

        # --- NEW: Intelligent Weight Initialization ---
        # Select the best initialization method based on the activation function.
        if self.activation_name in ["relu", "leaky_relu"]:
            # He Initialization is best for the ReLU family.
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif self.activation_name == "sigmoid":
            # Xavier/Glorot Initialization is best for Sigmoid and Tanh.
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
            # A fallback for linear or other activation functions.
            self.weights = np.random.randn(input_size, output_size) * 0.01

        self.bias = np.zeros((1, output_size))
        
        # Assign the activation functions.
        activations = {
            "sigmoid": (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            "relu": (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            "leaky_relu": (
                lambda x: ActivationFunctions.leaky_relu(x, alpha=0.01),
                lambda x: ActivationFunctions.leaky_relu_derivative(x, alpha=0.01)
            ),
            "linear": (ActivationFunctions.linear, ActivationFunctions.linear_derivative)
        }

        if self.activation_name not in activations:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
        self.activation_func, self.activation_derivative_func = activations[self.activation_name]
        
        # Cache for backpropagation.
        self.input = None
        self.output = None
        self.delta_weights = None
        self.delta_bias = None

    def forward(self, input_data):
        """Performs the forward pass through the layer."""
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.activation_func(self.output)
    
    def backward(self, output_error):
        """
        Performs the backward pass.
        Calculates weight/bias gradients and returns the input error for the previous layer.
        """
        activated_error = output_error * self.activation_derivative_func(self.output)
        self.delta_weights = np.dot(self.input.T, activated_error)
        self.delta_bias = np.sum(activated_error, axis=0, keepdims=True)
        input_error = np.dot(activated_error, self.weights.T)
        return input_error
    
class NeuralNetwork:
    """A fully-connected feedforward neural network."""
    def __init__(self):
        """Initializes the network components."""
        self.layers = []
        self.loss_func = None
        self.optimizer = None

    def add_layer(self, input_size, output_size, activation_name="sigmoid"):
        """Adds a new layer to the network."""
        self.layers.append(Layer(input_size, output_size, activation_name))

    def compile(self, optimizer, loss):
        """Configures the model for training with a specified optimizer and loss function."""
        self.optimizer = optimizer
        self.loss_func = loss

    def forward(self, input_data):
        """Propagates input data through all layers."""
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, y_pred):
        """Initiates the backpropagation process starting from the loss derivative."""
        error = self.loss_func.derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            error = layer.backward(error)

    @staticmethod
    def calculate_accuracy(y_pred, y_true):
        """Calculates classification accuracy for binary problems."""
        predicted_labels = np.round(y_pred)
        return np.mean(predicted_labels == y_true)

# NeuralNetwork sınıfının içinde

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size=32, verbose=True):
        """
        Trains the neural network using mini-batch gradient descent.
        """
        if self.optimizer is None or self.loss_func is None:
            raise ValueError("Network must be compiled before training.")
        
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle the training data at the beginning of each epoch
            permutation = np.random.permutation(num_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            epoch_loss = []
            epoch_acc = []

            # Loop over Mini-batches
            for i in range(0, num_samples, batch_size):
                # Get the current mini-batch
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                # 1. Forward pass
                train_preds = self.forward(X_batch)
                
                # 2. Backward pass
                self.backward(y_batch, train_preds)
                
                # 3. Update weights 
                for layer in self.layers:
                    self.optimizer.update(layer)
                
                # Record metrics for this batch
                epoch_loss.append(self.loss_func.loss(y_batch, train_preds))
                epoch_acc.append(self.calculate_accuracy(train_preds, y_batch))

            # Calculate average metrics for the epoch
            history['loss'].append(np.mean(epoch_loss))
            history['accuracy'].append(np.mean(epoch_acc))
            
            # Calculate validation metrics 
            val_preds = self.forward(X_val)
            history['val_loss'].append(self.loss_func.loss(y_val, val_preds))
            history['val_accuracy'].append(self.calculate_accuracy(val_preds, y_val))

            if verbose and epoch % (epochs // 10 or 1) == 0:
                print(f"Epoch {epoch}/{epochs} -> "
                    f"Loss: {history['loss'][-1]:.4f}, Acc: {history['accuracy'][-1]:.4f} | "
                    f"Val_Loss: {history['val_loss'][-1]:.4f}, Val_Acc: {history['val_accuracy'][-1]:.4f}")
        
        return history

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_history(history, title=''):
    """
    Plots training and validation accuracy/loss in two separate subplots.
    """
    plt.style.use('fast')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    fig.suptitle(f'Eğitim Geçmişi: {title}', fontsize=16)

    ax1.set_title('Model Doğruluğu')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Doğruluk')
    ax1.plot(history['accuracy'], label='Eğitim Doğruluğu')
    ax1.plot(history['val_accuracy'], label='Validasyon Doğruluğu')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Model Kaybı')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Kayıp')
    ax2.plot(history['loss'], label='Eğitim Kaybı')
    ax2.plot(history['val_loss'], label='Validasyon Kaybı')
    ax2.legend()
    ax2.grid(True)

    num_epochs = len(history['loss'])
    for ax in [ax1, ax2]:
        ax.set_xlim(0, num_epochs - 1)
        ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Prepare the dataset (XOR Problem)
    # For this simple problem, we use the same data for training and validation.
    X_data = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_data = np.array([[0], [1], [1], [0]])

    # 2. Build the Neural Network
    nn = NeuralNetwork()
    nn.add_layer(input_size=2, output_size=8, activation_name="leaky_relu")
    nn.add_layer(input_size=8, output_size=1, activation_name="sigmoid")
    
    # 3. Compile the model
    nn.compile(optimizer=SGD(learning_rate=0.1), loss=BinaryCrossEntropy())

    # 4. Train the model
    print("Eğitim başlıyor...")
    training_history = nn.train(X_train=X_data, y_train=y_data, 
                                X_val=X_data, y_val=y_data, 
                                epochs=500)
    print("Eğitim tamamlandı.")

    # 5. Visualize the results
    plot_history(training_history, title='XOR Problemi Çözümü')