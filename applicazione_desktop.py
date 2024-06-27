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
    img = img.resize((128, 128))  # Ridimensiona l'immagine a 128x128 pixel
    img = np.array(img) / 255.0 # Normalizza l'immagine
    img = np.expand_dims(img, axis=0) # Aggiunge una dimensione per l'input del modello
    return img

# Funzione per fare la predizione
def predict(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)[0] # Esegue la predizione sul modello
    top_indices = predictions.argsort()[-3:][::-1] # Ottiene le tre classi con la più alta probabilità
    #argsort --> restituisce gli elementi in ordine crescente
    #-3: --> seleziona gli ultimi 3 indici della lista, cioè le classi con probabilità più alta
    #::-1 --> inverte l'ordine degli elementi selezionati (prob. più alta a prob. più bassa)
    top_classes = top_indices
    top_probabilities = predictions[top_indices]
    return list(zip(top_classes, top_probabilities))

# Funzione per caricare l'immagine e fare la predizione
def load_image():
    global top_predictions
    file_path = filedialog.askopenfilename() #apre la finestra di dialogo per aprire un file
    if file_path:
        img = Image.open(file_path)     
        img = img.resize((250, 250))  # Ridimensiona per la visualizzazione
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk) #mostra l'immagine nella GUI
        panel.image = img_tk

        # Fai la predizione
        top_predictions = predict(file_path)
        update_display() #Aggiorno la visuale dei risultati

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

# Aggiungere uno sfondo
#background_image = Image.open("background.jpg")
#background_image = background_image.resize((800, 600), Image.ANTIALIAS)
#background_photo = ImageTk.PhotoImage(background_image)

#background_label = tk.Label(root, image=background_photo)
#background_label.place(relwidth=1, relheight=1)

# Titolo dell'applicazione
title = tk.Label(root, text="Identificatore di Galassie", font=("Helvetica", 24), bg="lightblue")
title.pack(pady=10)

panel = tk.Label(root)
panel.pack(side="top", fill="both", expand="yes", pady=20)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 14), bg="lightblue")
result_label.pack(side="bottom", fill="both", expand="yes", pady=20)

load_button = tk.Button(root, text="Carica Immagine", command=load_image, font=("Helvetica", 12), bg="lightgreen")
load_button.pack(side="bottom", fill="both", expand="yes", pady=10)

display_mode = tk.StringVar(value='Probabilità')
switch_button = tk.Button(root, text="Switch View", command=switch_view, font=("Helvetica", 12), bg="lightyellow")
switch_button.pack(side="bottom", fill="both", expand="yes", pady=10)

# Crediti
credits = tk.Label(root, text="credits: Esa/Nasa Hubble", font=("Helvetica", 10), bg="lightblue")
credits.pack(side="bottom", pady=5)

root.mainloop()
