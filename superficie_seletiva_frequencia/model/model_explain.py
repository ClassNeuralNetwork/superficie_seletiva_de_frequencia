import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import shap

model = tf.keras.models.load_model('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\model\\model.keras')

input_test = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\input_test_standard.csv')
output_test = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\output_test.csv')
names = pd.read_csv('C:\\Users\\lucas\\Downloads\\superficie_seletiva_de_frequencia\\superficie_seletiva_frequencia\\dataset\\test\\input_test.csv')

explainer = shap.KernelExplainer(model.predict, input_test)
shap_values = explainer(input_test)

# valor de frequencia de ressonancia
shap.summary_plot(shap_values[:, :, 0], input_test, plot_type="bar", feature_names=names.columns)

#valor de largura de banda
shap.summary_plot(shap_values[:, :, 1], input_test, plot_type="bar", feature_names=names.columns)


# valor de frequencia de ressonancia
shap.summary_plot(shap_values[:, :, 0], input_test, feature_names=names.columns)

#valor de largura de banda
shap.summary_plot(shap_values[:, :, 1], input_test, feature_names=names.columns)