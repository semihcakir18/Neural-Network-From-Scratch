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

# --- Kayıp Fonksiyonları (Loss Functions) ---
class Loss:
    """Kayıp fonksiyonları için temel sınıf."""
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    """Ortalama Kare Hatası (Mean Squared Error) kaybı."""
    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class BinaryCrossEntropy(Loss):
    """İkili Çapraz Entropi (Binary Cross-Entropy) kaybı."""
    def loss(self, y_true, y_pred):
        # log(0) hatasını önlemek için tahminleri kırp
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

# --- Optimize Edici Sınıfı (Optimizer Class) ---
class SGD:
    """Stokastik Gradyan İnişi (Stochastic Gradient Descent) optimize edicisi."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        """Katmanın ağırlıklarını ve bias'larını güncelle."""
        layer.weights -= self.learning_rate * layer.delta_weights
        layer.bias -= self.learning_rate * layer.delta_bias

# --- Katman Sınıfı (Layer Class) ---
class Layer:
    def __init__(self, input_size, output_size, activation_name="sigmoid"):
        # He ve Xavier/Glorot başlatma yöntemleri daha iyi sonuçlar verebilir
        # ancak şimdilik basitliği koruyoruz.
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))

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

    def backward(self, output_error):
        # Bu metod artık ağırlıkları GÜNCELLEMİYOR, sadece gradyanları HESAPLIYOR.
        activated_error = output_error * self.activation_derivative_func(self.output)

        self.delta_weights = np.dot(self.input.T, activated_error)
        self.delta_bias = np.sum(activated_error, axis=0, keepdims=True)

        input_error = np.dot(activated_error, self.weights.T)
        return input_error

# --- Ağ Sınıfı (Network Class) ---
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_func = None
        self.optimizer = None

    def add_layer(self, input_size, output_size, activation_name="sigmoid"):
        self.layers.append(Layer(input_size, output_size, activation_name))
        
    def compile(self, optimizer, loss):
        """Ağın eğitim için yapılandırılması (Keras'tan esinlenilmiştir)."""
        self.optimizer = optimizer
        self.loss_func = loss

    def forward(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true, y_pred):
        # Geri yayılımı kayıp fonksiyonunun türeviyle başlat
        error = self.loss_func.derivative(y_true, y_pred)
        
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def train(self, X_train, y_train, epochs):
        if self.optimizer is None or self.loss_func is None:
            raise ValueError("Ağ kullanılmadan önce 'compile' edilmelidir.")
            
        for epoch in range(epochs):
            # 1. İleri Yayılım (Forward Pass)
            predictions = self.forward(X_train)
            
            # 2. Kaybı Hesapla (Calculate Loss)
            loss = self.loss_func.loss(y_train, predictions)
            
            # 3. Geri Yayılım (Backward Pass - Gradyanları Hesapla)
            self.backward(y_train, predictions)
            
            # 4. Ağırlıkları Güncelle (Update Weights)
            for layer in self.layers:
                self.optimizer.update(layer)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

# --- Kullanım Örneği ---
if __name__ == "__main__":
    # Basit XOR problemi için veri seti
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]]) # XOR çıktısı

    # Ağ oluşturma 
    nn = NeuralNetwork()

    # Katmanları ekleme
    nn.add_layer(input_size=2, output_size=4, activation_name="relu")
    nn.add_layer(input_size=4, output_size=1, activation_name="sigmoid")

    # YENİ ADIM: Modeli derleme
    # Sınıflandırma için BinaryCrossEntropy daha iyi bir seçimdir.
    # Öğrenme oranı artık Optimizer içinde belirtiliyor.
    nn.compile(optimizer=SGD(learning_rate=0.1), loss=BinaryCrossEntropy())

    # Modeli eğitme
    print("Eğitim başlıyor...")
    nn.train(X, y, epochs=5000)
    print("Eğitim tamamlandı.")

    # Tahmin yapma
    print("\nTahminler:")
    predictions = nn.forward(X)
    for i in range(len(X)):
        print(f"Giriş: {X[i]}, Tahmin: {predictions[i][0]:.4f} -> {np.round(predictions[i][0])}, Gerçek: {y[i][0]}")