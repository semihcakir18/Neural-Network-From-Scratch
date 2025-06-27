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


# --- Adam Optimizer Class ---
class Adam:
    """Adam (Adaptive Moment Estimation) optimizer."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Initializes the Adam optimizer parameters."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestep counter, initialized to 0

    def update(self, layer):
        """Updates the layer's weights and biases using the Adam algorithm."""

        # 1. Increment the timestep
        self.t += 1

        # --- Update Weights ---
        # Get gradients for the weights
        grad_w = layer.delta_weights

        # Update first moment estimate (m)
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * grad_w

        # Update second moment estimate (v)
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (grad_w**2)

        # Compute bias-corrected first and second moment estimates
        m_hat_w = layer.m_weights / (1 - self.beta1**self.t)
        v_hat_w = layer.v_weights / (1 - self.beta2**self.t)

        # Update the weights
        layer.weights -= (
            self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        )

        # --- Update Biases ---
        # Get gradients for the biases
        grad_b = layer.delta_bias

        # Update first moment estimate (m)
        layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * grad_b

        # Update second moment estimate (v)
        layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (grad_b**2)

        # Compute bias-corrected first and second moment estimates
        m_hat_b = layer.m_bias / (1 - self.beta1**self.t)
        v_hat_b = layer.v_bias / (1 - self.beta2**self.t)

        # Update the biases
        layer.bias -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)


# ---------------------------------------------------------------------------
# Network Architecture
# ---------------------------------------------------------------------------


# --- Layer Class ---
class Layer:
    """Represents a single dense layer in the neural network."""

    def __init__(self, input_size, output_size, activation_name="sigmoid"):
        """Initializes the layer's weights, biases, and activation function."""

        self.activation_name = activation_name

        # --- Weight Initialization ---
        if self.activation_name in ["relu", "leaky_relu"]:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                2.0 / input_size
            )
        elif self.activation_name == "sigmoid":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(
                1.0 / input_size
            )
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.01

        self.bias = np.zeros((1, output_size))

        # --- NEW: Initialize Adam optimizer variables for each parameter ---
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)

        # Assign activation functions
        activations = {
            "sigmoid": (
                ActivationFunctions.sigmoid,
                ActivationFunctions.sigmoid_derivative,
            ),
            "relu": (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            "leaky_relu": (
                lambda x: ActivationFunctions.leaky_relu(x, alpha=0.01),
                lambda x: ActivationFunctions.leaky_relu_derivative(x, alpha=0.01),
            ),
            "linear": (
                ActivationFunctions.linear,
                ActivationFunctions.linear_derivative,
            ),
        }
        if self.activation_name not in activations:
            raise ValueError(f"Unsupported activation function: {self.activation_name}")
        self.activation_func, self.activation_derivative_func = activations[
            self.activation_name
        ]

        self.input = None
        self.output = None
        self.delta_weights = None
        self.delta_bias = None

    def forward(self, input_data, training=True):
        """
        Performs the forward pass through the layer.
        The 'training' argument is accepted for API consistency but not used here.
        """
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.activation_func(self.output)

    def backward(self, output_error):
        activated_error = output_error * self.activation_derivative_func(self.output)
        self.delta_weights = np.dot(self.input.T, activated_error)
        self.delta_bias = np.sum(activated_error, axis=0, keepdims=True)
        input_error = np.dot(activated_error, self.weights.T)
        return input_error


# --- Dropout Layer Class ---
class DropoutLayer:
    """
    A layer that applies Dropout regularization to prevent overfitting.
    This layer has no trainable parameters.
    """

    def __init__(self, rate):
        """
        Initializes the Dropout layer.
        Args:
            rate (float): The fraction of neurons to drop (e.g., 0.5 for 50%).
        """
        self.rate = rate  # The probability of dropping a neuron
        self.mask = None  # Will store the dropout mask for backpropagation

    def forward(self, input_data, training=True):
        """
        Performs the forward pass for the Dropout layer.
        """
        if not training:
            # During prediction/validation, do nothing.
            return input_data

        # During training, apply dropout.
        # 1. Create a binary mask where 1s indicate neurons to keep.
        # 2. Scale the mask by (1.0 - self.rate) -> this is "inverted dropout".
        self.mask = np.random.binomial(1, 1.0 - self.rate, size=input_data.shape) / (
            1.0 - self.rate
        )

        # Apply the mask to the input.
        return input_data * self.mask

    def backward(self, output_error):
        """
        Performs the backward pass for the Dropout layer.
        The gradient is simply passed through the same mask.
        """
        return output_error * self.mask


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

    def add_dropout_layer(self, rate):
        """A helper method to add a dropout layer for clarity."""
        self.layers.append(DropoutLayer(rate))

    # --- UPDATED forward method ---
    def forward(self, input_data, training=True):
        """
        Propagates input data through all layers.
        The 'training' flag is crucial for layers like Dropout.
        """
        output = input_data
        for layer in self.layers:
            # Pass the training flag to the layer's forward method
            output = layer.forward(output, training=training)
        return output

    # --- UPDATED train method ---
    def train(
        self, X_train, y_train, X_val, y_val, epochs, batch_size=32, verbose=True
    ):
        """Trains the neural network using mini-batch gradient descent."""
        if self.optimizer is None or self.loss_func is None:
            raise ValueError("Network must be compiled before training.")

        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_train_shuffled, y_train_shuffled = (
                X_train[permutation],
                y_train[permutation],
            )

            epoch_loss, epoch_acc = [], []

            for i in range(0, num_samples, batch_size):
                X_batch, y_batch = (
                    X_train_shuffled[i : i + batch_size],
                    y_train_shuffled[i : i + batch_size],
                )

                # 1. Forward pass in TRAINING mode
                train_preds = self.forward(X_batch, training=True)

                # 2. Backward pass
                self.backward(y_batch, train_preds)

                # 3. Update weights (only for layers that have them)
                for layer in self.layers:
                    if hasattr(layer, "weights"):  # Check if the layer is trainable
                        self.optimizer.update(layer)

                epoch_loss.append(self.loss_func.loss(y_batch, train_preds))
                epoch_acc.append(self.calculate_accuracy(train_preds, y_batch))

            # Calculate average metrics for the epoch
            history["loss"].append(np.mean(epoch_loss))
            history["accuracy"].append(np.mean(epoch_acc))

            # 4. Forward pass on validation data in PREDICTION mode
            val_preds = self.forward(X_val, training=False)
            history["val_loss"].append(self.loss_func.loss(y_val, val_preds))
            history["val_accuracy"].append(self.calculate_accuracy(val_preds, y_val))

            if verbose and epoch % (epochs // 10 or 1) == 0:
                print(
                    f"Epoch {epoch}/{epochs} -> "
                    f"Loss: {history['loss'][-1]:.4f}, Acc: {history['accuracy'][-1]:.4f} | "
                    f"Val_Loss: {history['val_loss'][-1]:.4f}, Val_Acc: {history['val_accuracy'][-1]:.4f}"
                )

        return history


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_history(history, title=""):
    """
    Plots training and validation accuracy/loss in two separate subplots.
    """
    plt.style.use("fast")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle(f"Eğitim Geçmişi: {title}", fontsize=16)

    ax1.set_title("Model Doğruluğu")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Doğruluk")
    ax1.plot(history["accuracy"], label="Eğitim Doğruluğu")
    ax1.plot(history["val_accuracy"], label="Validasyon Doğruluğu")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Model Kaybı")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Kayıp")
    ax2.plot(history["loss"], label="Eğitim Kaybı")
    ax2.plot(history["val_loss"], label="Validasyon Kaybı")
    ax2.legend()
    ax2.grid(True)

    num_epochs = len(history["loss"])
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
    X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_data = np.array([[0], [1], [1], [0]])

    # 2. Build the Neural Network with Dropout
    nn = NeuralNetwork()
    nn.add_layer(input_size=2, output_size=16, activation_name="relu")
    nn.add_dropout_layer(rate=0.1)
    nn.add_layer(input_size=16, output_size=1, activation_name="sigmoid")

    # 3. Compile the model
    nn.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossEntropy())

    # 4. Train the model
    print("Eğitim başlıyor...")
    training_history = nn.train(
        X_train=X_data,
        y_train=y_data,
        X_val=X_data,
        y_val=y_data,
        epochs=2000,
        batch_size=4,
    )  # Using batch_size for mini-batch demo
    print("Eğitim tamamlandı.")

    # 5. Visualize the results
    plot_history(training_history, title="XOR with Dropout")
