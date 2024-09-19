import cv2
import numpy as np
import os
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Görüntülerin yüklendiği klasör yolları
folder_paths = [
    {"path": "/content/drive/MyDrive/archive/Car-Bike-Dataset/Bike", "label": 0},
    {"path": "/content/drive/MyDrive/archive/Car-Bike-Dataset/Car", "label": 1}
]

data= []
labels = []

for  folder in folder_paths:
    files = os.listdir(folder["path"])
    random.shuffle(files)  # Dosyaları karıştır
    count = 0  # Seçilen resim sayısını tutmak için sayaç
    for filename in files:
        if filename.endswith(('.jpg', '.png', '.jpeg')) and count < 200 :
            img = cv2.imread(os.path.join(folder["path"], filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV ile renk formatını düzeltme
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (224, 224))  # Resmi boyutlandırma
            data.append(img)
            labels.append(folder["label"])
            count += 1

# Seçilen resimlerin gösterimi
num_samples = len(data)  # Seçilen resim sayısı
rows = int(np.ceil(num_samples / 10))  # Satır sayısını hesapla
fig, axes = plt.subplots(rows, 10, figsize=(20, 2 * rows))  # Daha büyük bir genişlik ve azaltılmış yükseklik ayarla
for i, ax in enumerate(axes.flatten()):
    if i < num_samples:
        ax.imshow(data[i])
        #ax.set_title('Label: ' + ('Car' if labels[i] == 1 else 'Bike'))  # Resmin etiketini başlığa ekle
        ax.axis('off')
    else:
        ax.axis('off')  # Kullanılmayan subplot'ları gizle

plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Subplotlar arası boşlukları minimize et
plt.show()


# Giriş verilerini tekrarlayarak üç kanallı hale getirme
X = np.array(data) / 255.0
y = np.array(labels)

# Verileri eğitim ve test seti olarak ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# VGG16 modelini yükleme
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Model tanımı ve eğitimi
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Modelin üst katmanlarını eğitmek
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleme
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Erken durdurma callback'i tanımlama
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Modeli eğitme
history = model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_test, Y_test))

# evaluate_and_visualize_model fonksiyonu
def evaluate_and_visualize_model(model, X, y, num_samples):
    predictions = model.predict(X)
    predicted_labels = (predictions > 0.5).astype(int)

    num_columns = 10  # Satır başına 10 resim
    rows = int(np.ceil(num_samples / num_columns))
    fig, axes = plt.subplots(rows, num_columns, figsize=(20, 2 * rows))  # Genişlik artırıldı, yükseklik azaltıldı
    axes = axes.flatten()  # Eksenleri tek boyuta indirge

    for i, ax in enumerate(axes):
        if i < len(X):
            ax.imshow(X[i].reshape(224, 224, 3), interpolation='nearest')  # VGG16 için resim boyutu
            predicted_label = 'Car' if predicted_labels[i] == 1 else 'Bike'
            actual_label = 'Car' if y[i] == 1 else 'Bike'
            # Başlık yerine etiketleri kullanarak, resmin üzerine yazmayacak şekilde ayarla
            ax.text(1, 1, f'Prediction: {predicted_label}\nTrue: {actual_label}', color='black', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
            ax.axis('off')
        else:
            ax.axis('off')  # Kullanılmayan subplot'ları gizle

    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Subplotlar arası boşlukları azalt
    plt.show()

# Modelin performansını değerlendirme ve görselleştirme
evaluate_and_visualize_model(model, X, y, len(X))

# Eğitim ve doğrulama kaybını ve başarı puanını grafik üzerinde göster
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



# Test seti üzerinde modelin nihai başarı puanını ve kaybını yazdır
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Final Test Loss: {loss:.4f}")
print(f"Final Test Accuracy: {accuracy:.4f}")

def plot_confusion_matrix(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

# Modelin tahminlerini al ve ikili eşik uygula
predictions = model.predict(X_test)
predicted_labels = (predictions > 0.5).astype(int)

# Confusion matrix ve classification report'u görselleştir ve yazdır
plot_confusion_matrix(Y_test, predicted_labels)
print_classification_report(Y_test, predicted_labels)