import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import shap



input_test = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\input_test_standard.csv')
output_test = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\output_test.csv')

history = pd.read_csv('loss.csv')

model = tf.keras.models.load_model('model.keras')


test_loss, test_mae = model.evaluate(input_test, output_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

y_pred = model.predict(input_test)

# Calcular as métricas para cada saída do modelo
print('')

print('MSE - RF: ', mean_squared_error(output_test['resonant_frequency(GHZ)'], y_pred[:, 0]))
print('R2 - RF: ', r2_score(output_test['resonant_frequency(GHZ)'], y_pred[:, 0]))
print('')
print('MSE - BW: ', mean_squared_error(output_test['BW(GHZ)'], y_pred[:, 1]))
print('R2 - BW: ', r2_score(output_test['BW(GHZ)'], y_pred[:, 1]))

# Calcular métricas de avaliação
mae_rf = mean_absolute_error(output_test['resonant_frequency(GHZ)'], y_pred[:, 0])
mae_bw = mean_absolute_error(output_test['BW(GHZ)'], y_pred[:, 1])

print('')
print(f'MAE - RF: {mae_rf}')
print(f'MAR - BW: {mae_bw}')

# Plotar a perda durante o treinamento
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='validation')
plt.legend()
#plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()
