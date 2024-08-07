








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

explainer = shap.Explainer(model.predict, input_test)

shap_values = (input_test).values
shap.summary_plot(shap_values, input_test, plot_type="bar", feature_names=names.columns)
shap.summary_plot(shap_values, input_test, feature_names=names.columns)

shap.dependence_plot("2", shap_values, input_test)