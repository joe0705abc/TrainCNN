import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. 載入 MNIST 資料集（Keras 自帶）
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 資料前處理：正規化並調整維度
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# 3. 建立 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # MNIST 共10類數字
])

# 4. 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 訓練模型
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 6. 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ 測試準確率：{test_acc:.4f}")

# 7. 預測並顯示部分結果
predictions = model.predict(x_test[:10])

for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"預測: {predictions[i].argmax()}, 正確: {y_test[i]}")
    plt.axis('off')
    plt.show()
