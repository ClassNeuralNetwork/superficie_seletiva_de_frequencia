# Criação do modelo


import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#-----------------------------------------------------


input_train = pd.read_csv('superficie_seletiva_frequencia\dataset\train\input_train.csv.csv')
output_train = pd.read_csv('superficie_seletiva_frequencia\dataset\train\output_train.csv')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, input_dim=4,activation='sigmoid' ))
model.add(tf.keras.layers.Dropout(0.2)) 
model.add(tf.keras.layers.Dense(units=64, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.2)) 

model.add(tf.keras.layers.Dense(units=2))

# Resumo do modelo
model.summary()