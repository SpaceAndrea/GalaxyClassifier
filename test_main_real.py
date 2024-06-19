
#Step 1
#IMPORTARE LE LIBRERIE NECESSARIE

#Al momento importo tutto le librerie che negli esercizi
#fatti in classe abbiamo importato, successivamente valuterò
#quali tenere e quali non mi servono.
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras 
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.api.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
from keras.src.models import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
#from tensorflow.keras.optimizers import Adam
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

###############################################################

#STEP 2:
#Caricare e preprocessare il dataset
# Percorsi ai dataset
train_images_path = 'data/images_training_rev1/images_training_rev1'
test_images_path = 'data/images_test_rev1/images_test_rev1'
train_labels_path = 'solutions/training_solutions_rev1/training_solutions_rev1.csv'

# Caricare le etichette di training
train_labels = pd.read_csv(train_labels_path)

# Aggiungere il percorso completo alle immagini nel DataFrame
train_labels['GalaxyID'] = train_labels['GalaxyID'].apply(lambda x: os.path.join(train_images_path, f'{x}.jpg'))

# Verificare se i file esistono
missing_files = [path for path in train_labels['GalaxyID'] if not os.path.exists(path)]
print(f"Missing files: {len(missing_files)}")

if len(missing_files) > 0:
    print(missing_files[:10])  # Stampare alcuni percorsi di file mancanti

# Rimuovere i file mancanti dal DataFrame
train_labels = train_labels[train_labels['GalaxyID'].apply(os.path.exists)]

print(train_labels.head())

# Definire il generatore di immagini con normalizzazione e divisione per validazione
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Generatore di immagini per il training set
train_generator = datagen.flow_from_dataframe(
    dataframe=train_labels,
    x_col='GalaxyID',
    y_col=train_labels.columns[1:].tolist(),
    target_size=(128, 128),
    batch_size=32,
    class_mode='raw',
    subset='training'
)

# Generatore di immagini per il validation set
validation_generator = datagen.flow_from_dataframe(
    dataframe=train_labels,
    x_col='GalaxyID',
    y_col=train_labels.columns[1:].tolist(),
    target_size=(128, 128),
    batch_size=32,
    class_mode='raw',
    subset='validation'
)

# Definire il modello
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    #Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(37, activation='softmax')  # Assumendo che ci siano 37 classi
])

# Compilare il modello
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Visualizzare il sommario del modello
model.summary()

# Addestrare il modello
history = model.fit(
    train_generator,
    epochs=10,  # numero di epoche, puoi cambiarlo
    validation_data=validation_generator
)

# Valutare il modello
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

# Salvare il modello
model.save('galaxy_classifier_model.h5')

# Salvare la cronologia dell'addestramento
pd.DataFrame(history.history).to_csv('training_history.csv', index=False)

# Visualizzare i risultati
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()


# Definire il generatore di immagini per il test set
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Caricare le immagini di test
test_files = [os.path.join(test_images_path, fname) for fname in os.listdir(test_images_path)]
test_df = pd.DataFrame({'GalaxyID': test_files})

# Generatore di immagini per il test set
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='GalaxyID',
    target_size=(128, 128),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

# Fare predizioni sulle immagini di test
predictions = model.predict(test_generator)

# Salvare le predizioni in un file CSV
predictions_df = pd.DataFrame(predictions, columns=train_labels.columns[1:])
predictions_df.insert(0, 'GalaxyID', test_df['GalaxyID'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]))
predictions_df.to_csv('test_predictions.csv', index=False)


# Caricare i benchmark
benchmark_ones = pd.read_csv('benchmark/all_ones_benchmark/all_ones_benchmark.csv')
benchmark_zeros = pd.read_csv('benchmark/all_zeros_benchmark/all_zeros_benchmark.csv')
benchmark_central_pixel = pd.read_csv('benchmark/central_pixel_benchmark/central_pixel_benchmark.csv')

# Funzione per calcolare l'accuratezza di un benchmark
def calculate_accuracy(predictions, benchmark):
    return np.mean(np.argmax(predictions.values[:, 1:], axis=1) == np.argmax(benchmark.values[:, 1:], axis=1))

# Caricare le predizioni del test
test_predictions = pd.read_csv('test_predictions.csv')

# Calcolare e confrontare le accuratezze
accuracy_ones = calculate_accuracy(test_predictions, benchmark_ones)
accuracy_zeros = calculate_accuracy(test_predictions, benchmark_zeros)
accuracy_central_pixel = calculate_accuracy(test_predictions, benchmark_central_pixel)

print(f'Accuracy with All Ones Benchmark: {accuracy_ones}')
print(f'Accuracy with All Zeros Benchmark: {accuracy_zeros}')
print(f'Accuracy with Central Pixel Benchmark: {accuracy_central_pixel}')
