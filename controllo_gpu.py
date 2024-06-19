
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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#Suggerimenti per ridurre il tempo di addestramento
#Ridurre il dataset: Se possibile, usa un subset del tuo dataset per velocizzare i test iniziali.
#Batch size più grande: Un batch size più grande può ridurre il numero di batch per epoca, ma richiede più memoria.
#Utilizzare GPU: Se hai accesso a una GPU, utilizza TensorFlow per sfruttare l'accelerazione hardware.
#Ottimizzare il modello: Ridurre la complessità del modello (meno layer o meno unità per layer).