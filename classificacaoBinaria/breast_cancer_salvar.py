import numpy as np
import pandas as pd
import keras
from keras.models import Sequential #classe responsavel por criar a rede neural com varias camadas ocultas
from keras.layers import Dense, Dropout #classe responsavel por adicionar camadas densas a rede neural (cada um dos neuronios é ligado com todos os neuronios da camada subsequente)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv') # classsificação binaria 1 = tem tumor, 0 não tem

classificador = Sequential()
classificador.add(Dense(units = 8, activation= 'relu',
                        kernel_initializer= 'normal', input_dim = 30))
classificador.add(Dropout(0.2)) #zera 20% valores dos neuronios da camada

classificador.add(Dense(units = 8, activation= 'relu',
                        kernel_initializer= 'normal'))

classificador.add(Dropout(0.2)) #zera 20% valores dos neuronios da camada

classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=10, epochs=100)


classificador_json = classificador.to_json()

with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('classificador_breast.h5')