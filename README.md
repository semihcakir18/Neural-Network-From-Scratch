# Neural Network from Scratch

A simple feedforward neural network built from the ground up using only Python and NumPy. This project is an educational exercise to understand the core mechanics of deep learning without relying on high-level frameworks like TensorFlow or PyTorch.

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project implements a fully-connected neural network with the fundamental components required for training and prediction. The code is structured in a modular, object-oriented way to clearly separate responsibilities, making it easy to extend and experiment with.

## Key Features

- **Modular Design:** The network is built from distinct `Layer`, `Optimizer`, and `Loss` objects, mimicking the architecture of modern deep learning libraries.
- **Forward & Backward Propagation:** Core implementation of the backpropagation algorithm to train the network.
- **Activation Functions:** Includes standard activations (`Sigmoid`, `ReLU`, `Linear`) with their derivatives.
- **Loss Functions:** Abstracted `Loss` class with `Mean Squared Error (MSE)` and `Binary Cross-Entropy (BCE)` implementations.
- **Optimizers:** An `Optimizer` class with `Stochastic Gradient Descent (SGD)` as the first implementation.
- **Zero Dependencies (Almost!):** Built entirely on `NumPy` for all numerical operations.

## Requirements

- Python 3.x
- NumPy

## How to Use

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**

    ```bash
    pip install numpy
    ```

3.  **Run the example:**
    The script includes a self-contained example that trains the network to solve the classic XOR problem. Simply run the Python file:
    ```bash
    python your_file_name.py
    ```

### Example Code

The `if __name__ == "__main__"` block demonstrates how to build, compile, and train the network:

```python
if __name__ == "__main__":
    # Basit XOR problemi için veri seti
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]]) # XOR çıktısı

    # 1. Ağ oluşturma
    nn = NeuralNetwork()

    # 2. Katmanları ekleme
    nn.add_layer(input_size=2, output_size=4, activation_name="relu")
    nn.add_layer(input_size=4, output_size=1, activation_name="sigmoid")

    # 3. Modeli derleme (optimize edici ve kayıp fonksiyonu ile)
    nn.compile(optimizer=SGD(learning_rate=0.1), loss=BinaryCrossEntropy())

    # 4. Modeli eğitme
    print("Eğitim başlıyor...")
    nn.train(X, y, epochs=5000)
    print("Eğitim tamamlandı.")

    # 5. Tahmin yapma
    print("\nTahminler:")
    predictions = nn.forward(X)
    for i in range(len(X)):
        print(f"Giriş: {X[i]}, Tahmin: {predictions[i][0]:.4f} -> {np.round(predictions[i][0])}, Gerçek: {y[i][0]}")
```
