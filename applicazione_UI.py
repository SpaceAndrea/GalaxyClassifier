import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Carica il modello salvato
model = tf.keras.models.load_model('galaxy_model.keras')

# Descrizioni delle classi
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

# Funzione per preprocessare l'immagine
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # Dimensione che il modello si aspetta
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Funzione per fare la predizione
def predict(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    top_classes = top_indices
    top_probabilities = predictions[top_indices]
    return list(zip(top_classes, top_probabilities))

# Funzione per caricare l'immagine e fare la predizione
def load_image():
    global top_predictions
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((250, 250))  # Ridimensiona per la visualizzazione
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Fai la predizione
        top_predictions = predict(file_path)
        update_display()

# Funzione per aggiornare la visualizzazione
def update_display():
    if display_mode.get() == 'Probabilità':
        result_text.set(
            f"Top 3 Predizioni:\n"
            f"1. Classe: {top_predictions[0][0]}, Probabilità: {top_predictions[0][1]:.2f}\n"
            f"2. Classe: {top_predictions[1][0]}, Probabilità: {top_predictions[1][1]:.2f}\n"
            f"3. Classe: {top_predictions[2][0]}, Probabilità: {top_predictions[2][1]:.2f}"
        )
    else:
        result_text.set(
            f"Top 3 Predizioni:\n"
            f"1. Classe: {top_predictions[0][0]}, Descrizione: {class_descriptions[top_predictions[0][0]]}\n"
            f"2. Classe: {top_predictions[1][0]}, Descrizione: {class_descriptions[top_predictions[1][0]]}\n"
            f"3. Classe: {top_predictions[2][0]}, Descrizione: {class_descriptions[top_predictions[2][0]]}"
        )

# Funzione per commutare la visualizzazione
def switch_view():
    if display_mode.get() == 'Probabilità':
        display_mode.set('Descrizioni')
    else:
        display_mode.set('Probabilità')
    update_display()

# Crea l'interfaccia grafica
root = tk.Tk()
root.title("Galaxy Classifier")

panel = tk.Label(root)
panel.pack(side="top", fill="both", expand="yes")

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text)
result_label.pack(side="bottom", fill="both", expand="yes")

load_button = tk.Button(root, text="Carica Immagine", command=load_image)
load_button.pack(side="bottom", fill="both", expand="yes")

display_mode = tk.StringVar(value='Probabilità')
switch_button = tk.Button(root, text="Switch View", command=switch_view)
switch_button.pack(side="bottom", fill="both", expand="yes")

root.mainloop()
