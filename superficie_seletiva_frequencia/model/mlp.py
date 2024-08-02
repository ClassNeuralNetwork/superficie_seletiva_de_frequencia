import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
#-----------------------------------------------------

input_train = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\train\\input_train.csv')
output_train = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\train\\output_train.csv')

input_test = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\input_test.csv')
output_test = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\output_test.csv')

# Normalizar os dados
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train) 
print(input_train)
input_test = scaler.transform(input_test)

# Criar e compilar o modelo
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, input_dim=4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=128, input_dim=4, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=256, activation='sigmoid'))
model.add(tf.keras.layers.Dropout(0.2)) 
model.add(tf.keras.layers.Dense(units=2))

# Resumo do modelo
model.summary()

opt = Adam(learning_rate=0.01)
# opt = SGD(learning_rate=0.01)

model.compile(optimizer=opt, loss='mse', metrics=['mae'])

# Definir a callback de parada precoce
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Treinar o modelo com parada precoce
history = model.fit(input_train, output_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
print(history)

# Plotar a perda durante o treinamento
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

# Prever os valores usando input_train
y_pred = model.predict(input_test)
print(y_pred)

print('')
# Calcular as métricas para cada saída do modelo
print('MSE - 1st output: ', mean_squared_error(output_test['resonant_frequency(GHZ)'], y_pred[:, 0]))
print('R2 - 2st output: ', r2_score(output_test['resonant_frequency(GHZ)'], y_pred[:, 0]))

print('')

print('MSE - 2st output: ', mean_squared_error(output_test['BW(GHZ)'], y_pred[:, 1]))
print('R2 - 2st output: ', r2_score(output_test['BW(GHZ)'], y_pred[:, 1]))

# ------------ RESULTADOS ------------ #

# MSE - 1st output:  0.7862020204136747
# R2 - 2st output:  0.9835229048525664

# MSE - 2st output:  0.14781390388633842
# R2 - 2st output:  0.953245025459346