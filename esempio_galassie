import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.src.models import Sequential
from keras.src.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.src.callbacks import history
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.src.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt


# Percorsi ai dataset
train_images_path = 'data/images_training_rev1/images_training_rev1'
test_images_path = 'data/images_test_rev1/images_test_rev1'
train_labels_path = 'solutions/training_solutions_rev1/training_solutions_rev1.csv'

def load_images_from_folder(folder, image_size=(64, 64), limit=None):
    images = []
    image_ids = []
    for filename in os.listdir(folder)[:limit]:
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img = img.resize(image_size)
            img = np.array(img) / 255.0
            images.append(img)
            image_id = int(filename.split('.')[0])
            image_ids.append(image_id)
    return np.array(images), np.array(image_ids)

# Caricamento delle immagini di addestramento
train_images, train_image_ids = load_images_from_folder(train_images_path, image_size=(64, 64), limit=10000)

# Caricamento delle etichette di addestramento
train_labels_df = pd.read_csv(train_labels_path)
train_labels = []
for image_id in train_image_ids:
    train_labels.append(train_labels_df[train_labels_df['GalaxyID'] == image_id].iloc[:, 1:].values[0])

train_labels = np.array(train_labels)

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

# Divisione dei dati in training e validation set
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

print("imploso?")
# Addestramento del modello
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Caricamento delle immagini di test
test_images, test_image_ids = load_images_from_folder(test_images_path, image_size=(64, 64), limit=10000)

# Predizione delle etichette per il test set
predictions = model.predict(test_images)

# Creazione del DataFrame con le predizioni
predictions_df = pd.DataFrame(predictions, columns=['Class' + str(i) for i in range(predictions.shape[1])])
predictions_df.insert(0, 'GalaxyID', test_image_ids)

# Salvataggio delle predizioni in un file CSV
predictions_df.to_csv('galaxies_predictions.csv', index=False)

# Mappatura delle classi alle loro descrizioni
class_descriptions = {
    0: "La galassia appare liscia, senza particolari caratteristiche.",
    1: "La galassia ha caratteristiche visibili o un disco.",
    2: "La galassia è identificata come stella o artefatto.",
    3: "La galassia è vista di taglio.",
    4: "La galassia non è vista di taglio.",
    5: "La galassia ha una barra visibile.",
    6: "La galassia non ha una barra visibile.",
    7: "La galassia ha una struttura a spirale.",
    8: "La galassia non ha una struttura a spirale.",
    9: "La galassia ha una forma rotonda.",
    10: "La galassia ha una forma squadrata.",
    11: "La galassia non ha un rigonfiamento centrale.",
    12: "La galassia ha un disco visibile di taglio.",
    13: "La galassia non ha un disco visibile di taglio.",
    14: "La galassia ha un rigonfiamento centrale arrotondato.",
    15: "La galassia ha un rigonfiamento centrale squadrato.",
    16: "La galassia non ha un rigonfiamento centrale.",
    17: "La galassia ha un braccio a spirale.",
    18: "La galassia ha due bracci a spirale.",
    19: "La galassia ha tre bracci a spirale.",
    20: "La galassia ha quattro bracci a spirale.",
    21: "La galassia ha cinque bracci a spirale.",
    22: "La galassia ha sei bracci a spirale.",
    23: "La galassia ha sette bracci a spirale.",
    24: "La galassia ha otto bracci a spirale.",
    25: "La galassia ha nove bracci a spirale.",
    26: "La galassia ha dieci bracci a spirale.",
    27: "La galassia ha un rigonfiamento centrale arrotondato.",
    28: "La galassia ha un rigonfiamento centrale squadrato.",
    29: "La galassia non ha un rigonfiamento centrale.",
    30: "La galassia ha una struttura ad anello.",
    31: "La galassia non ha una struttura ad anello.",
    32: "La galassia ha una struttura a lente o arco.",
    33: "La galassia non ha una struttura a lente o arco.",
    34: "La galassia è risultata da una fusione.",
    35: "La galassia non è risultata da una fusione.",
    36: "La galassia è sovrapposta con altre galassie o stelle."
}

# Visualizzazione delle immagini di test con le classificazioni
def display_predictions(images, image_ids, predictions_df, num_images=10):
    plt.figure(figsize=(20, 15))
    for i in range(min(num_images, len(images))):
        ax = plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"GalaxyID: {image_ids[i]}")
    
        
        # Trova la classificazione con la probabilità più alta
        galaxy_id = image_ids[i]
        prediction_row = predictions_df[predictions_df['GalaxyID'] == galaxy_id]
        prediction_row = prediction_row.drop(columns=['GalaxyID']).values[0]
        top_class = np.argmax(prediction_row)
        top_class_prob = prediction_row[top_class]
        
        description = class_descriptions[top_class]
        
        plt.xlabel(f"Class: {top_class}, Prob: {top_class_prob:.2f}")
        plt.ylabel(description)
        plt.axis("off")

    plt.show()

# Visualizzazione delle immagini di test con le predizioni
display_predictions(test_images, test_image_ids, predictions_df, num_images=10)

# Creazione del file di testo con le descrizioni
with open('galaxies_descriptions.txt', 'w') as file:
    for i in range(len(test_images)):
        galaxy_id = test_image_ids[i]
        prediction_row = predictions_df[predictions_df['GalaxyID'] == galaxy_id]
        prediction_row = prediction_row.drop(columns=['GalaxyID']).values[0]
        top_class = np.argmax(prediction_row)
        top_class_prob = prediction_row[top_class]
        description = class_descriptions[top_class]
        file.write(f"GalaxyID: {galaxy_id}, Class: {top_class}, Prob: {top_class_prob:.2f}, Description: {description}\n")

print("File 'galaxies_descriptions.txt' creato con successo.")