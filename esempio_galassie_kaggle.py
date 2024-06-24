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
train_images_path = 'data/images_training_rev1/images_training_rev1'
test_images_path = 'data/images_test_rev1/images_test_rev1'
train_labels_path = 'solutions/training_solutions_rev1/training_solutions_rev1.csv'

# Percorsi ai dataset sul portatile
# train_images_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/images_training_rev1/images_training_rev1'
# test_images_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/images_test_rev1/images_test_rev1'
# train_labels_path = 'C:/Users/AndreaBianchini/Downloads/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1/training_solutions_rev1.csv' 

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
#Modello CNN (rete neurale convoluzionale)
model = Sequential([
#Spiegazione:
#Kernel: Un kernel (o filtro) è una matrice 2D (in questo caso 3x3) che scorre sull'immagine di input e calcola il prodotto scalare tra il kernel e la regione dell'immagine coperta dal kernel.
#Padding: Se non viene utilizzato il padding (cioè, il padding è 'valid'), la dimensione dell'output si riduce rispetto all'input. Con un kernel 3x3, ogni convoluzione riduce la dimensione dell'immagine di 2 (1 pixel da ogni lato). Quindi, l'immagine di dimensione 64x64 diventa 62x62.
#Numero di Filtri: Qui specifichiamo che vogliamo 32 filtri, quindi otteniamo 32 feature maps. Ogni filtro è responsabile dell'estrazione di diverse caratteristiche dall'immagine di input (bordo, texture, ecc.).
    #Conv2D Layer (32 filtri, kernel 3x3, ReLU):
    #Input: immagine di dimensione (64, 64, 3), cioè 64x64 pixel con 3 canali colori.
    #Output: Un set di 32 feature maps (mappe di caratteristiche), ciascuna di dimensioni (62, 62) ***Non ho ben capito***
    #Funzione di attivazione: ReLU introduce non-linearità.
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    #MaxPooling2D Layer (pooling 2x2):
    #Input: Le 32 feature maps di dimensioni (62, 62) dall'output del livello precedente (Conv2D).
    #Output: 32 feature maps ridotte a dimensioni (31, 31) attraverso l'operazione di pooling.
    #Operazione: Prende il massimo valore in ogni finestra 2x2 ***Non ho ben capito***
    MaxPooling2D((2, 2)),
    #Conv2D Layer (64 filtri, kernel 3x3, ReLU):
    #Input: Le 32 feature maps di dimensioni (31, 31) (output precedente).
    #Output: Un set di 64 feature maps, ciascuna di dimensioni (29, 29).
    #Funzione di Attivazione: ReLU.
    Conv2D(64, (3, 3), activation='relu'),
    #MaxPooling2D Layer (pooling 2x2):
    #Input: Le 64 feature maps di dimensioni (29, 29) (output precedente).
    #Output: 64 feature maps ridotte a dimensioni (14, 14).
    MaxPooling2D((2, 2)),
    #Flatten Layer:
    #Input: Le 64 feature maps di dimensioni (14, 14) (output precedente).
    #Output: Un vettore unidimensionale (1D) di lunghezza 12544 (64 * 14 * 14).
    Flatten(),
    #Funzione: Disattiva casualmente il 50% dei neuroni durante l'addestramento per prevenire l'overfitting.
    Dropout(0.5),
    #Dense Layer (128 neuroni, ReLU):
    #Input: Il vettore 1D di lunghezza 12544.
    #Output: Un vettore 1D di lunghezza 128.
    #Funzione di Attivazione: ReLU.
    Dense(128, activation='relu'),
    #Output Layer (37 neuroni, sigmoid):
    #Input: Il vettore 1D di lunghezza 128.
    #Output: Un vettore 1D di lunghezza 37, con valori tra 0 e 1.
    #Funzione di Attivazione: Sigmoid (permette la regressione multilabel).
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

#ipotetico:

# Salvataggio dello storico dell'allenamento
#with open('history.json', 'w') as f:
    #json.dump(history.history, f)
#print("Storico dell'allenamento salvato in 'history.json'")

# Carica lo storico dell'allenamento
#with open('history.json', 'r') as f:
    #history = json.load(f)

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

#Per caricarlo in futuro:
model = tf.keras.models.load_model('galaxy_model.keras')

# ---- Aggiungi questa parte per fare le predizioni sui dati di test ---- #

# Caricamento delle immagini di test
test_files = os.listdir(test_images_path)
test_image_paths = [os.path.join(test_images_path, file) for file in test_files]

# Creazione del dataset tf.data per le immagini di test
test_dataset = create_dataset(test_image_paths, labels=None)

# Predizione delle etichette per il test set
predictions = model.predict(test_dataset, verbose=1)

# Creazione del DataFrame con le predizioni
test_filenames = [os.path.basename(path) for path in test_image_paths]
predictions_df = pd.DataFrame(predictions, columns=[f'Class{i}' for i in range(predictions.shape[1])])
predictions_df.insert(0, 'GalaxyID', [int(f.split('.')[0]) for f in test_filenames])

# Salvataggio delle predizioni in un file CSV
predictions_df.to_csv('galaxies_predictions.csv', index=False)

print("Predizioni salvate in 'galaxies_predictions.csv'")