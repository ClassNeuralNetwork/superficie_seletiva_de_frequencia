import pandas as pd
from sklearn.model_selection import train_test_split

db = pd.read_csv("superficie_seletiva_frequencia\dataset\dataset.csv")

lista = ['resonant_frequency(GHZ)','BW(GHZ)','h','p','d','w']
db.columns = lista

db.head()
db.info()
classes = db[['resonant_frequency(GHZ)','BW(GHZ)']].value_counts()
print(classes)

input_train, input_test, output_train, output_test = train_test_split(db.iloc[:, 2:5].values, db[['resonant_frequency(GHZ)','BW(GHZ)']].values, test_size=0.2)

input_train = pd.DataFrame(input_train)
input_train.to_csv('input_train_csv', index=False)

input_test = pd.DataFrame(input_test)
input_test.to_csv('input_test_csv', index=False)

output_train = pd.DataFrame(output_train)
output_train.to_csv('output_train_csv', index=False)

output_test = pd.DataFrame(output_test)
output_train.to_csv('output_test_csv', index=False)

input_train = pd.read_csv('input_train_csv')
output_train = pd.read_csv('output_train_csv')
input_train.head()