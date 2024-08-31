import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
import shap
from sklearn.metrics import mean_squared_error, r2_score
#-----------------------------------------------------

input_train = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\train\\input_train.csv')
output_train = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\train\\output_train.csv')
input_test = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\input_test.csv')
output_test = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\output_test.csv')

# Normalizar os dados
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train) 
input_test = scaler.transform(input_test)


# Salvando os dados normalizados
pd.DataFrame(input_train).to_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\train\\input_train_standard.csv', index=False)
pd.DataFrame(input_test).to_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\input_test_standard.csv', index=False)
                               


# Criar e compilar o modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=32, input_dim=4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=32, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=128, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.22)) 
model.add(tf.keras.layers.Dense(units=2))

# Resumo do modelo
model.summary()

opt = Adam(learning_rate=0.019)

# opt = SGD(learning_rate=0.01)

model.compile(optimizer=opt, loss='mse', metrics=['mae'])

# Definir a callback de parada precoce
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Treinar o modelo com parada precoce
history = model.fit(input_train, output_train, epochs=250, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# salvar o history
pd.DataFrame(history.history).to_csv('loss.csv', index=False)

model.save('model.keras')