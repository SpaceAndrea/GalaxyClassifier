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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from keras.src.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt
from keras.src.utils import image_dataset_from_directory
import json
from keras.src.regularizers import L2

# # Percorsi ai dataset
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

#In sintesi:
    #Carica l'immagine dal percorso specificato (img_path).
    #Decodifica l'immagine come JPEG con 3 canali (RGB).
    #Ridimensiona l'immagine a 64x64 pixel.
    #Normalizza i valori dei pixel tra 0 e 1.
    #Restituisce l'immagine preprocessata e l'array delle etichette.

def load_image(img_path, label): #prende in input il path dell'immagine e l'etichetta associata(label)
    img = tf.io.read_file(img_path) #memorizzo nella variabile 'img' il percorso dell'immagine come tensore grezzo di byte
    img = tf.image.decode_jpeg(img, channels=3) #decodifica l'immagine jpeg nel tensore 'img' specificando che l'immagine ha 3 canali (RGB)
    img = tf.image.resize(img, [128, 128]) #ridimensiona l'immagine in 64x64 pixel
    img = img / 255.0  # Normalizzazione dei pixel dell'immagine per essere compresi tra 0 e 1 
    return img, label #restituisco l'immagine preprocessata e l'etichetta associata

#In sintesi:
    #Crea un dataset TensorFlow dai percorsi delle immagini (image_paths) e dalle etichette (labels).
    #Applica la funzione load_image a ciascun elemento del dataset in parallelo.
    #Mescola il dataset con un buffer di 1000 elementi.
    #Raggruppa gli elementi del dataset in batch di dimensione 64.
    #Precarica i dati per migliorare l'efficienza.
    #Restituisce il dataset preprocessato.

def create_dataset(image_paths, labels): #prende in input il path delle immagini e le etichette associate
    dataset = Dataset.from_tensor_slices((image_paths, labels)) #crea un dataset tensorflow dai tensori 'image_paths' e 'labels' 
    #ed ogni elemento del dataset è una coppia
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE) #applica la funzione load_image a ogni elemento del dataset.
    #tf.data.AUTOTUNE consente a tensorflow di determinare automaticamente il numero ottimale di thread di esecuzione per migliorare perf.
    dataset = dataset.shuffle(buffer_size=1000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)
    #dataset.shuffle(buffer_size)=1000 --> mescola gli elementi del dataset con un buffer di 1000 elementi
    #dataset.batch(64) --> raccoglie gli elementi del dataset in batch di dimensione 64
    #dataset.prefetch(buffer_size=tf.data.AUTOTUNE) --> precarica i dati nel buffer per garantire che la GPU abbia sempre i dati pronti
    #per l'addestramento, migliorando l'efficenza
    return dataset #la funzione restituisce il dataset preprocessato


# Creazione del DataFrame con i percorsi delle immagini e le etichette
#Creo un elenco image_paths, contenente i percorsi completi delle immagini di training
image_paths = [os.path.join(train_images_path, f"{int(img_id)}.jpg") for img_id in train_labels_df['GalaxyID']]
#Estraggo le etichette associate alle immagini dal Dataframe del csv solutions training, escludendo la colonna del GalaxyID
labels = train_labels_df.iloc[:, 1:].values #seleziono tutte le colonne tranne la prima e le converto in un array numpy

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
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
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
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=[early_stopping])
# Salvataggio del modello
model.save('galaxy_model.keras')
print("Modello salvato in 'galaxy_model.keras'")
# Salvataggio dello storico dell'allenamento
with open('history.json', 'w') as f:
    json.dump(history.history, f)
print("Storico dell'allenamento salvato in 'history.json'")

# Carica lo storico dell'allenamento
with open('history.json', 'r') as f:
    history = json.load(f)

# Plotting training & validation accuracy values
plt.figure(figsize=(6, 4))

# Sotto-grafico per l'accuratezza
plt.plot(history['accuracy'], label='Train')
plt.plot(history['val_accuracy'], label='Validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.figure(figsize=(6, 4))

# Sotto-grafico per la perdita
plt.plot(history['loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

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

#Carico il mio test prediction
test_predictions = pd.read_csv('galaxies_predictions.csv')

#Carico i benchmark
#Benchmark con soli 1:
benchmark_ones = pd.read_csv('benchmark/all_ones_benchmark/all_ones_benchmark.csv')
#Benchmark con soli 0:
benchmark_zeros = pd.read_csv('benchmark/all_zeros_benchmark/all_zeros_benchmark.csv')
#Benchmark che valuta il pixel centrale:
benchmark_central_pixel = pd.read_csv('benchmark/central_pixel_benchmark/central_pixel_benchmark.csv')

# Funzione per calcolare l'accuratezza di un benchmark
def calculate_accuracy(predictions, benchmark):
    return np.mean(np.argmax(predictions.values[:, 1:], axis=1) == np.argmax(benchmark.values[:, 1:], axis=1))

# Calcolo l'accuratezza paragonando le mie prediction ai benchmark
accuracy_ones = calculate_accuracy(test_predictions, benchmark_ones)
accuracy_zeros = calculate_accuracy(test_predictions, benchmark_zeros)
accuracy_central_pixel = calculate_accuracy(test_predictions, benchmark_central_pixel)

print(f'Accuracy with All Ones Benchmark: {accuracy_ones}')
print(f'Accuracy with All Zeros Benchmark: {accuracy_zeros}')
print(f'Accuracy with Central Pixel Benchmark: {accuracy_central_pixel}')
# significa che il 62.38% delle predizioni totali del modello (sia positive che negative) corrispondono a quelle del benchmark Central Pixel.

# Calcoli generali
y_true = np.argmax(benchmark_central_pixel.values[:, 1:], axis=1)
y_pred = np.argmax(test_predictions.values[:, 1:], axis=1)

# Precision, Recall, F1 Score
precision = precision_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'Precision: {precision}') #implica che quasi tutte le galassie che il tuo modello ha predetto come positive sono effettivamente positive secondo il benchmark Central Pixel.
print(f'F1 Score: {f1}') # rappresenta un bilancio tra precisione e recall, suggerendo che il modello è abbastanza bilanciato ma potrebbe migliorare nel catturare tutte le istanze positive.