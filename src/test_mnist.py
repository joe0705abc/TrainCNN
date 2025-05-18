from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 載入模型（如果已儲存）
model = tf.keras.models.load_model("mnist_cnn_model.h5")


# 預處理函式 + 回傳原圖與處理後圖
def preprocess_image(image_path):
    original = Image.open(image_path)
    gray = original.convert('L')
    resized = gray.resize((28, 28))

    img_array = np.array(resized) / 255.0
    inverted = 1 - img_array
    final = inverted.reshape(1, 28, 28, 1)

    return original, resized, inverted, final


# 測試圖片
image_path = "C:/Users/User/PycharmProjects/TrainCNN/9.jpg"
orig, resized, inverted, processed = preprocess_image(image_path)

# 顯示原圖、灰階縮圖、預處理圖
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(orig)
axes[0].set_title("原始圖片")
axes[0].axis("off")

axes[1].imshow(resized, cmap='gray')
axes[1].set_title("灰階 + 縮放 28x28")
axes[1].axis("off")

axes[2].imshow(inverted, cmap='gray')
axes[2].set_title("預處理後（反轉 + 正規化）")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# 預測
prediction = model.predict(processed)
print("預測結果：", prediction.argmax())
