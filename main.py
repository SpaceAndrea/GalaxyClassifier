
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
#CARICARE IL DATASET
#Ho messo il dataset nella cartella data; la quale è già
#suddivisa in:
#-images_test_rev1: contiene 61578 immagini jpg di galassie
#-images_training_rev1: contiene 79975 immagini jpg di galassie

#le immagini non sono divise in classi, dentro le cartelle
#ci sono immagini e basta.
#Si potrebbe forse dividerle in cartelle? Ma in quel caso
#bisogna valutare come.

# Percorsi ai dataset
#Path computer fisso:
train_images_path = 'data/images_training_rev1'
test_image_path = 'data/images_test_rev1'
train_labels_path = 'solutions/training_solutions_rev1/training_solutions_rev1.csv'
#Path computer ITS:


# Caricare le etichette
train_labels = pd.read_csv(train_labels_path)


# Aggiungere il percorso completo alle immagini nel DataFrame
train_labels['GalaxyID'] = train_labels['GalaxyID'].apply(lambda x: os.path.join(train_images_path, f'{x}.jpg'))
#Questa lambda prende un argomento x (che è l'ID di una galassia)
# e restituisce il percorso completo al file immagine 
# corrispondente utilizzando os.path.join per combinare 
# train_images_path e x con l'estensione .jpg.

# Quindi, se train_images_path è data/images_training_rev1 
# e un GalaxyID è 123456, la funzione lambda genererà il percorso 
# data/images_training_rev1/123456.jpg.


# Verificare se i file esistonox
print(train_labels.head())

###############################################################

#STEP 3: UTILIZZARE ImageDataGenerator PER LA NORMALIZZAZIONE
# E LA DIVISIONE DEI DATI

#ImageDataGenerator carica e preprocessa le immagini in modo 
# dinamico durante l'addestramento. Questo significa che non tutte
# le immagini devono essere caricate in memoria contemporaneamente, 
# il che è particolarmente utile quando si lavora con dataset di 
# grandi dimensioni.

# Definire il generatore di immagini con normalizzazione e divisione per validazione
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
print(datagen)

#Usare flow_from_dataframe con validation_split 
# semplifica il processo di suddivisione del dataset in training 
# e validation set. Non devi gestire manualmente la suddivisione 
# dei dati, la normalizzazione e la preparazione dei batch.

# Generatore di immagini per il training set
train_generator = datagen.flow_from_dataframe(
    dataframe=train_labels,
    x_col='GalaxyID',
    y_col=train_labels.columns[1:].tolist(),
    target_size=(128, 128),  # mantenere la risoluzione originale
    batch_size=32,
    class_mode='raw',  # poiché le etichette sono distribuzioni di probabilità
    subset='training'
)

#dataframe=train_labels: Usa il DataFrame train_labels per ottenere i percorsi delle immagini e le etichette.
#x_col='GalaxyID': Colonna del DataFrame che contiene i percorsi delle immagini.
#y_col=train_labels.columns[1:].tolist(): Colonne del DataFrame che contengono le etichette (distribuzioni di probabilità).
#target_size=(128, 128): Dimensioni a cui ridimensionare tutte le immagini.
#batch_size=32: Numero di immagini per batch.
#class_mode='raw': Utilizza le etichette esattamente come sono nel DataFrame.
#subset='training': Indica che questo generatore sarà utilizzato per il training set.

#flow_from_dataframe: Questo metodo consente di creare generatori 
# di immagini direttamente da un DataFrame di pandas, che può 
# essere molto utile quando si ha un dataset in cui i percorsi 
# delle immagini e le relative etichette sono memorizzati in un 
# DataFrame.

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

###############################################################

#STEP 4: DEFINIRE IL MODELLO

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
    Dropout(0.5),
    Dense(37, activation='softmax')  # Assumendo che ci siano 37 classi
])

# Compilare il modello
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Visualizzare il sommario del modello
model.summary()

###############################################################

#STEP 5: ADDESTRAMENTO DEL MODELLO

# Addestrare il modello
history = model.fit(
    train_generator,
    epochs=10,  # numero di epoche, puoi cambiarlo
    validation_data=validation_generator
)

###############################################################

#STEP 6: VALUTAZIONE DEL MODELLO E SALVATAGGIO

# Valutare il modello
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

# Salvare il modello
model.save('galaxy_classifier_model.h5')

# Salvare la cronologia dell'addestramento
pd.DataFrame(history.history).to_csv('training_history.csv', index=False)

###############################################################

#STEP 7: VISUALIZZARE I RISULTATI

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

###############################################################

#STEP 8: MATRICE DI CONFUSIONE

# Predizioni sul validation set
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calcolare la matrice di confusione
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()