import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.src.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.src.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt

# Percorsi ai dataset
#train_images_path = 'data/images_training_rev1/images_training_rev1'
#test_images_path = 'data/images_test_rev1/images_test_rev1'
#train_labels_path = 'solutions/training_solutions_rev1/training_solutions_rev1.csv'

# Percorsi ai dataset sul portatile
train_images_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/images_training_rev1/images_training_rev1'
test_images_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/images_test_rev1/images_test_rev1'
train_labels_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1/training_solutions_rev1.csv' 

'''
test_image_files = os.listdir(test_images_path)
print(f"Numero di immagini di test: {len(test_image_files)}")

# Funzione per caricare le etichette di addestramento
def load_labels(train_labels_path):
    train_labels_df = pd.read_csv(train_labels_path)
    return train_labels_df

# Generatore di dati
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Caricamento delle etichette
train_labels_df = load_labels(train_labels_path)

# Funzione per creare il DataFrame delle etichette di addestramento
def create_label_dataframe(folder, labels_df):
    files = os.listdir(folder)
    data = []
    for file in files:
        image_id = int(file.split('.')[0])
        label = labels_df[labels_df['GalaxyID'] == image_id].iloc[:, 1:].values[0]
        data.append((file, *label))
    columns = ['filename'] + [f'Class{i}' for i in range(labels_df.shape[1] - 1)]
    return pd.DataFrame(data, columns=columns)

# Creazione del DataFrame delle etichette
train_labels_df = create_label_dataframe(train_images_path, train_labels_df)

# Generatori di immagini
train_generator = datagen.flow_from_dataframe(
    train_labels_df,
    directory=train_images_path,
    x_col='filename',
    y_col=train_labels_df.columns[1:].tolist(),
    target_size=(64, 64),
    batch_size=32,
    class_mode='raw',
    subset='training')

validation_generator = datagen.flow_from_dataframe(
    train_labels_df,
    directory=train_images_path,
    x_col='filename',
    y_col=train_labels_df.columns[1:].tolist(),
    target_size=(64, 64),
    batch_size=32,
    class_mode='raw',
    subset='validation')

# Creazione del modello della rete neurale
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(37, activation='sigmoid')  # Utilizziamo sigmoid per una regressione multilabel
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)

# Addestramento del modello
model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks = early_stopping)

# Salvataggio del modello
model.save('galaxy_model.keras')
print("Modello salvato in 'galaxy_model.keras'")

'''
#Per caricarlo in futuro:
model = tf.keras.models.load_model('galaxy_model.keras')


##################################################################
#Parte di test

'''
# Creazione del DataFrame per le immagini di test
test_files = os.listdir(test_images_path)
test_data = [{'filename': file} for file in test_files]
test_df = pd.DataFrame(test_data)

# Generatore di dati
test_datagen = ImageDataGenerator(rescale=1./255)

# Caricamento delle immagini di test
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_images_path,
    x_col='filename',
    target_size=(64, 64),
    batch_size=32,
    class_mode=None,  # Nessuna etichetta per le immagini di test
    shuffle=False)

# Predizione delle etichette per il test set
predictions = model.predict(test_generator, verbose=1)

# Creazione del DataFrame con le predizioni
filenames = test_generator.filenames
predictions_df = pd.DataFrame(predictions, columns=[f'Class{i}' for i in range(predictions.shape[1])])
predictions_df.insert(0, 'GalaxyID', [int(f.split('.')[0]) for f in filenames])

# Salvataggio delle predizioni in un file CSV
predictions_df.to_csv('galaxies_predictions.csv', index=False)

print("Predizioni salvate in 'galaxies_predictions.csv'")

'''
# Plotting training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()