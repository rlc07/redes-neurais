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

novo = np.array([[15.80, 8.34, 118, 100, 40, 600, 400, 322, 16.09, 10,
                  11.30, 11, 13, 14, 17, 19, 190, 470, 99, 12,
                  132.90, 66.9, 324.90, 34, 12, 13, 15, 16, 18, 19]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)
