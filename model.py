import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Função para ajustar tamanho das imagens!
def ajustar_imagens(caminho, maximo_pixels):
    for filename in os.listdir(caminho):
        if filename.endswith(".jpg"):
            image_path = os.path.join(caminho, filename)
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            img.save(image_path, quality=85)
            width, height = img.size
            total_pixels = width * height
            
            if total_pixels > maximo_pixels:
                new_width = int((maximo_pixels / total_pixels) ** 0.5 * width)
                new_height = int((maximo_pixels / total_pixels) ** 0.5 * height)
                img = img.resize((new_width, new_height), Image.ANTIALIAS)
                img.save(image_path)


#localização do Dataset!
TRAINING_DIR = "dataset/train"
VALIDATION_DIR ="dataset/test"


maximo_pixels = 89478485  #limite de pixels!


ajustar_imagens(TRAINING_DIR, maximo_pixels)
ajustar_imagens(VALIDATION_DIR, maximo_pixels)

#Configurando modelo!

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),  
    MaxPooling2D(2, 2),  
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),  
    Dense(3, activation='softmax')#Número de classes definido para 3
]) 

#Otimizador do modelo e tipo de categoria
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#Preparando as imagens do dataset de treinamento
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))
 
 #Preparando as imagens do dataset de teste
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='categorical',
                                                              target_size=(150, 150))

#treinando modelo
history = model.fit(train_generator,
                              epochs=10,
                              verbose=1,
                              validation_data=validation_generator)

#Salvando o modelo
model.save("modelo.h5")

#Declarando variáveis dos resultados finais
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc)) 


#Mostrando O gráfico do modelo

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()