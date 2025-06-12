import numpy as np

# --- Aktivasyon Fonksiyonları ---
class ActivationFunctions:
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
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)

# --- Katman Sınıfı ---
class Layer:
    def __init__(self, input_size, output_size, activation_name="sigmoid"):
        self.weights = np.random.randn(input_size, output_size) * 0.01 # Ağırlıkları küçük rastgele değerlerle başlat
        self.bias = np.zeros((1, output_size)) # Bias'ları sıfırla başlat

        self.activation_name = activation_name
        if activation_name == "sigmoid":
            self.activation_func = ActivationFunctions.sigmoid
            self.activation_derivative_func = ActivationFunctions.sigmoid_derivative
        elif activation_name == "relu":
            self.activation_func = ActivationFunctions.relu
            self.activation_derivative_func = ActivationFunctions.relu_derivative
        elif activation_name == "linear":
            self.activation_func = ActivationFunctions.linear
            self.activation_derivative_func = ActivationFunctions.linear_derivative
        else:
            raise ValueError("Desteklenmeyen aktivasyon fonksiyonu.")

        # Geri yayılım için gerekli değerleri saklamak
        self.input = None
        self.output = None
        self.activated_output = None
        self.delta_weights = None
        self.delta_bias = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        self.activated_output = self.activation_func(self.output)
        return self.activated_output

    def backward(self, output_error, learning_rate):
        # Aktivasyon fonksiyonunun türevini uygula
        activated_error = output_error * self.activation_derivative_func(self.output)

        # Ağırlık ve bias gradyanlarını hesapla
        self.delta_weights = np.dot(self.input.T, activated_error)
        self.delta_bias = np.sum(activated_error, axis=0, keepdims=True)

        # Bir sonraki katmana iletmek için giriş hatasını hesapla
        input_error = np.dot(activated_error, self.weights.T)

        # Ağırlık ve bias'ları güncelle
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias

        return input_error

# --- Ağ Sınıfı ---
class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate

    def add_layer(self, input_size, output_size, activation_name="sigmoid"):
        self.layers.append(Layer(input_size, output_size, activation_name))

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, target, predictions):
        # Kayıp fonksiyonunun türevi (Basit MSE için)
        error = predictions - target # Bu, kayıp fonksiyonuna göre değişecektir
        
        # Katmanları tersten dolaşarak geri yayılımı yap
        for layer in reversed(self.layers):
            error = layer.backward(error, self.learning_rate)

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            predictions = self.forward(X_train)
            loss = np.mean(np.square(y_train - predictions)) # MSE kaybı
            self.backward(y_train, predictions)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- Kullanım Örneği ---
if __name__ == "__main__":
    # Basit XOR problemi için veri seti
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]]) # XOR çıktısı

    # Ağ oluşturma
    nn = NeuralNetwork(learning_rate=0.1)

    # Katmanları ekleme
    # Giriş katmanı (2 giriş nöronu), gizli katman (4 nöron, ReLU aktivasyonu)
    nn.add_layer(input_size=2, output_size=4, activation_name="relu")
    # Çıkış katmanı (1 nöron, sigmoid aktivasyonu)
    nn.add_layer(input_size=4, output_size=1, activation_name="sigmoid")

    # Modeli eğitme
    print("Eğitim başlıyor...")
    nn.train(X, y, epochs=5000)
    print("Eğitim tamamlandı.")

    # Tahmin yapma
    print("\nTahminler:")
    predictions = nn.forward(X)
    print(np.round(predictions)) # Yuvarlanmış tahminler
    print("\nGerçek Değerler:")
    print(y)