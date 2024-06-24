import tensorflow as tf
from tensorflow._api.v2.data import Dataset
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
from keras.src.utils import image_dataset_from_directory

# Percorsi ai dataset
#train_images_path = 'data/images_training_rev1/images_training_rev1'
#test_images_path = 'data/images_test_rev1/images_test_rev1'
#train_labels_path = 'solutions/training_solutions_rev1/training_solutions_rev1.csv'

'''
# Percorsi ai dataset sul portatile
train_images_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/images_training_rev1/images_training_rev1'
test_images_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/images_test_rev1/images_test_rev1'
train_labels_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1/training_solutions_rev1.csv' 

# Caricamento delle etichette
train_labels_df = pd.read_csv(train_labels_path)

# Funzione per creare il dataset tf.data
def load_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    img = img / 255.0  # Normalizzazione
    return img, label

def create_dataset(image_paths, labels):
    dataset = Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Creazione del DataFrame con i percorsi delle immagini e le etichette
image_paths = [os.path.join(train_images_path, f"{int(img_id)}.jpg") for img_id in train_labels_df['GalaxyID']]
labels = train_labels_df.iloc[:, 1:].values

# Split del dataset in training e validation
image_paths_train, image_paths_val, labels_train, labels_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Creazione dei dataset tf.data
train_dataset = create_dataset(image_paths_train, labels_train)
val_dataset = create_dataset(image_paths_val, labels_val)

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

model.summary()

# Callback per l'early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Addestramento del modello
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=[early_stopping])

# Salvataggio del modello
model.save('galaxy_model.keras')
print("Modello salvato in 'galaxy_model.keras'")

'''

#ipotetico:
'''
# Salvataggio dello storico dell'allenamento
with open('history.json', 'w') as f:
    json.dump(history.history, f)
print("Storico dell'allenamento salvato in 'history.json'")

# Carica lo storico dell'allenamento
with open('history.json', 'r') as f:
    history = json.load(f)
'''

#Per caricarlo in futuro:
history = model = tf.keras.models.load_model('galaxy_model.keras')

# Plotting training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()